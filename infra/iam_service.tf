# Two roles, because App Runner separates them:
#   - access role   : used by App Runner itself to pull the image from ECR
#   - instance role : assumed by the running container, for AWS calls it makes

data "aws_iam_policy_document" "apprunner_build_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_access" {
  name               = "${var.project}-apprunner-access"
  description        = "Lets App Runner pull the container image from ECR."
  assume_role_policy = data.aws_iam_policy_document.apprunner_build_assume.json
}

resource "aws_iam_role_policy_attachment" "apprunner_access_ecr" {
  role       = aws_iam_role.apprunner_access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}


data "aws_iam_policy_document" "apprunner_tasks_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["tasks.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "apprunner_instance" {
  name               = "${var.project}-apprunner-instance"
  description        = "Assumed by the running container."
  assume_role_policy = data.aws_iam_policy_document.apprunner_tasks_assume.json
}

data "aws_iam_policy_document" "apprunner_instance" {
  # Conversation history — scoped to this one table.
  statement {
    sid    = "HistoryTable"
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:DeleteItem",
    ]
    resources = [aws_dynamodb_table.history.arn]
  }

  # Secrets by exact ARN, not "*" — a wildcard here would let the container
  # read every secret in the account.
  statement {
    sid       = "ReadOwnSecrets"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.openai.arn, aws_secretsmanager_secret.api_key.arn]
  }
}

resource "aws_iam_role_policy" "apprunner_instance" {
  name   = "${var.project}-apprunner-instance"
  role   = aws_iam_role.apprunner_instance.id
  policy = data.aws_iam_policy_document.apprunner_instance.json
}
