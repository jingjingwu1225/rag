# Conversation history, stored outside the service so instances are fungible.
#
# This table is what lets the API scale horizontally at all: with history in
# an in-process checkpointer, turn 2 of a conversation landing on a different
# instance found nothing and silently answered without context.
resource "aws_dynamodb_table" "history" {
  name = "${var.project}-history"

  # On-demand: this workload is a handful of requests per demo, and
  # provisioned capacity would bill continuously for idle. Stays in the
  # free tier at this volume.
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "thread_id"

  attribute {
    name = "thread_id"
    type = "S"
  }

  # DynamoDB deletes expired items for free. Without this, every conversation
  # ever started is retained indefinitely.
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = false # demo data, reconstructible; PITR would add cost for no benefit
  }
}
