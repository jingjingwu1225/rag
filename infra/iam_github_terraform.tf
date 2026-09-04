# A second CI role, for Terraform only.
#
# Split from the deploy role deliberately. Terraform has to manage IAM, and a
# role that can rewrite IAM is the most dangerous credential in the account —
# so it is reachable only from infra.yml, which applies on explicit manual
# dispatch. The deploy role, which runs automatically on every push to main,
# can push images and read two secrets and nothing else.
resource "aws_iam_role" "github_terraform" {
  name               = "${var.project}-github-terraform"
  description        = "Assumed by the Infra workflow to run Terraform."
  assume_role_policy = data.aws_iam_policy_document.github_assume.json
}

data "aws_iam_policy_document" "github_terraform" {
  # Read-and-write over exactly the services this stack declares. Terraform
  # needs Describe/Get on every managed resource just to plan, so read and
  # write travel together here.
  statement {
    sid    = "ManageStackResources"
    effect = "Allow"
    actions = [
      "ecr:*",
      "dynamodb:CreateTable",
      "dynamodb:DeleteTable",
      "dynamodb:DescribeTable",
      "dynamodb:DescribeTimeToLive",
      "dynamodb:UpdateTimeToLive",
      "dynamodb:UpdateTable",
      "dynamodb:ListTagsOfResource",
      "dynamodb:TagResource",
      "dynamodb:DescribeContinuousBackups",
      "dynamodb:UpdateContinuousBackups",
      "secretsmanager:CreateSecret",
      "secretsmanager:DeleteSecret",
      "secretsmanager:DescribeSecret",
      "secretsmanager:TagResource",
      "secretsmanager:UpdateSecret",
      "secretsmanager:GetResourcePolicy",
      "apprunner:*",
      "budgets:*",
      "logs:*",
      "cloudwatch:*",
      "sts:GetCallerIdentity",
    ]
    resources = ["*"]
  }

  # IAM, scoped by name prefix so this role cannot touch unrelated principals
  # in the account — notably it cannot grant itself more privilege by editing
  # a role outside the rag-api-* namespace.
  statement {
    sid    = "ManageStackIam"
    effect = "Allow"
    actions = [
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:GetRole",
      "iam:PassRole",
      "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies",
      "iam:ListInstanceProfilesForRole",
      "iam:PutRolePolicy",
      "iam:GetRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
      "iam:TagRole",
      "iam:UpdateAssumeRolePolicy",
    ]
    resources = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.project}-*"]
  }

  # The OIDC provider is account-level and has no name-scoped ARN pattern
  # worth expressing; it is a single fixed resource.
  statement {
    sid    = "ManageOidcProvider"
    effect = "Allow"
    actions = [
      "iam:GetOpenIDConnectProvider",
      "iam:CreateOpenIDConnectProvider",
      "iam:DeleteOpenIDConnectProvider",
      "iam:UpdateOpenIDConnectProviderThumbprint",
      "iam:TagOpenIDConnectProvider",
    ]
    resources = [
      "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/token.actions.githubusercontent.com"
    ]
  }

  statement {
    sid    = "TerraformState"
    effect = "Allow"
    actions = [
      "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket",
    ]
    resources = [
      "arn:aws:s3:::rag-api-tfstate-${data.aws_caller_identity.current.account_id}",
      "arn:aws:s3:::rag-api-tfstate-${data.aws_caller_identity.current.account_id}/*",
    ]
  }
}

resource "aws_iam_role_policy" "github_terraform" {
  name   = "${var.project}-github-terraform"
  role   = aws_iam_role.github_terraform.id
  policy = data.aws_iam_policy_document.github_terraform.json
}

output "github_terraform_role_arn" {
  description = "Set as AWS_TF_ROLE_ARN in GitHub repo secrets."
  value       = aws_iam_role.github_terraform.arn
}
