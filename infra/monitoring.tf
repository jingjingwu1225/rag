# Alerting on the EMF metrics api.py already emits.
#
# Emitting metrics and alarming on them are different things: without alarms
# the dashboard only tells you something broke *after* you go looking. These
# are deliberately few — an alarm nobody acts on trains people to ignore the
# channel, so each one here corresponds to a thing worth waking up for.

resource "aws_sns_topic" "alerts" {
  name = "${var.project}-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.budget_alert_email
  # Note: AWS sends a confirmation email; the subscription stays
  # "PendingConfirmation" until the link is clicked. Terraform cannot
  # confirm it for you.
}

# No aws_cloudwatch_log_group here on purpose. App Runner creates and owns its
# log groups, and the name embeds a service ID that does not exist until the
# service does (/aws/apprunner/<name>/<service-id>/application). Declaring our
# own would create an empty group that nothing ever writes to, and any metric
# filter attached to it would silently never match. Retention on App Runner's
# group is set after the service exists.

# --- Latency -----------------------------------------------------------------
# A turn makes 5-12 OpenAI calls, so p99 in the tens of seconds is normal.
# This fires when it degrades past the point where App Runner's fixed ~120s
# request cap starts truncating real requests.
resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "${var.project}-high-latency"
  alarm_description   = "p99 answer latency approaching App Runner's 120s request cap."
  namespace           = "RagApi"
  metric_name         = "LatencyMs"
  extended_statistic  = "p99"
  period              = 300
  evaluation_periods  = 2 # two consecutive windows, so one slow burst is not an incident
  threshold           = 90000
  comparison_operator = "GreaterThanThreshold"
  # No data simply means no traffic on a demo service — not a failure.
  treat_missing_data = "notBreaching"
  alarm_actions      = [aws_sns_topic.alerts.arn]
  ok_actions         = [aws_sns_topic.alerts.arn]

  dimensions = { Service = var.project }
}

# --- Retrieval quality -------------------------------------------------------
# RetryCount rising means the grader is rejecting first-pass retrieval more
# often. That is a *quality* regression that never shows up as an error rate:
# the service keeps returning 200s, the answers just quietly get worse.
# This is the alarm that would catch a bad corpus or a prompt regression.
resource "aws_cloudwatch_metric_alarm" "retrieval_degraded" {
  alarm_name          = "${var.project}-retrieval-degraded"
  alarm_description   = "Self-correction loop retrying more than usual — retrieval quality may have regressed."
  namespace           = "RagApi"
  metric_name         = "RetryCount"
  statistic           = "Average"
  period              = 900
  evaluation_periods  = 2
  threshold           = 1.0 # avg >1 retry/turn; healthy baseline is ~0.2 (1 in 6)
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = { Service = var.project }
}

# --- Errors ------------------------------------------------------------------
# ServerErrors is emitted directly by api.py's middleware as EMF, rather than
# extracted by a log metric filter. Same reason as above: a filter needs a
# log group name, and App Runner's is not knowable until the service exists.
# Emitting from the app also keeps every metric in this namespace produced the
# same way, instead of some by the app and some by log parsing.
resource "aws_cloudwatch_metric_alarm" "server_errors" {
  alarm_name          = "${var.project}-server-errors"
  alarm_description   = "5xx responses from the API."
  namespace           = "RagApi"
  metric_name         = "ServerErrors"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 3
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  # Must match a dimension set the app actually publishes, or this alarm sits
  # in INSUFFICIENT_DATA forever while looking correctly configured.
  dimensions = { Service = var.project }
}

# --- Dashboard ---------------------------------------------------------------
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = var.project

  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6
        properties = {
          title  = "Latency (ms)"
          region = var.region
          metrics = [
            ["RagApi", "LatencyMs", "Service", "rag-api", { stat = "p50", label = "p50" }],
            ["...", { stat = "p99", label = "p99" }],
            ["RagApi", "RetrievalMs", "Service", "rag-api", { stat = "p50", label = "retrieval p50" }],
            ["RagApi", "GenerationMs", "Service", "rag-api", { stat = "p50", label = "generation p50" }],
          ]
          # Splitting retrieval from generation is what makes this actionable:
          # they have different causes and different fixes.
          view = "timeSeries"
        }
      },
      {
        type = "metric", x = 12, y = 0, width = 12, height = 6
        properties = {
          title  = "Retrieval quality"
          region = var.region
          metrics = [
            ["RagApi", "RetryCount", "Service", "rag-api", { stat = "Average", label = "avg retries/turn" }],
            ["RagApi", "ChunksRetrieved", "Service", "rag-api", { stat = "Average", label = "avg chunks" }],
            ["RagApi", "SubQueries", "Service", "rag-api", { stat = "Average", label = "avg sub-queries" }],
          ]
          view = "timeSeries"
        }
      },
      {
        type = "metric", x = 0, y = 6, width = 12, height = 6
        properties = {
          title  = "Traffic and errors"
          region = var.region
          metrics = [
            ["RagApi", "Turns", "Service", var.project, { stat = "Sum", label = "turns" }],
            ["RagApi", "ServerErrors", "Service", var.project, { stat = "Sum", label = "5xx" }],
          ]
          view = "timeSeries"
        }
      },
    ]
  })
}

output "dashboard_url" {
  description = "CloudWatch dashboard for the service."
  value       = "https://${var.region}.console.aws.amazon.com/cloudwatch/home?region=${var.region}#dashboards:name=${var.project}"
}
