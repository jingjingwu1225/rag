# GitHub Actions authenticates to AWS via OIDC — no long-lived access keys
# stored as repository secrets. GitHub mints a short-lived token per run and
# AWS exchanges it for temporary credentials.

data "aws_caller_identity" "current" {}

# Fetch GitHub's certificate chain at plan time instead of hardcoding a
# thumbprint. The value copied around in most examples (6938fd4d...) is a
# DigiCert root GitHub has since moved off; using it produces
# "Not authorized to perform sts:AssumeRoleWithWebIdentity", which reads like
# a trust-policy problem and sends you looking in the wrong place entirely.
# Certificates rotate — a hardcoded fingerprint is a scheduled outage.
data "tls_certificate" "github" {
  url = "https://token.actions.githubusercontent.com/.well-known/openid-configuration"
}

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  # Every cert in the chain (leaf, intermediate, root): AWS accepts up to 5,
  # and this avoids depending on which position the verified cert occupies.
  thumbprint_list = [for cert in data.tls_certificate.github.certificates : cert.sha1_fingerprint]
}

data "aws_iam_policy_document" "github_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # The load-bearing condition. Without scoping `sub`, ANY repository on
    # GitHub could assume this role — the OIDC provider vouches that the token
    # came from GitHub Actions, not that it came from *your* repo. Locked to
    # this repo's main branch specifically.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "repo:${var.github_repo}:ref:refs/heads/main",
        "repo:${var.github_repo}:pull_request",
      ]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "${var.project}-github-actions"
  description        = "Assumed by GitHub Actions via OIDC to build, push, and deploy."
  assume_role_policy = data.aws_iam_policy_document.github_assume.json
}

data "aws_iam_policy_document" "github_actions" {
  # Push images to this repository only.
  statement {
    sid     = "EcrAuth"
    effect  = "Allow"
    actions = ["ecr:GetAuthorizationToken"]
    # This one call is account-scoped by design; it returns only a token.
    resources = ["*"]
  }

  statement {
    sid    = "EcrPush"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "ecr:DescribeImages",
    ]
    resources = [aws_ecr_repository.api.arn]
  }

  # Read the OpenAI key so the eval gate can run against the built image.
  # Scoped to these two secrets by ARN — never "*".
  statement {
    sid       = "ReadSecrets"
    effect    = "Allow"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.openai.arn, aws_secretsmanager_secret.api_key.arn]
  }

  # Terraform state.
  statement {
    sid       = "TerraformState"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = ["arn:aws:s3:::rag-api-tfstate-${data.aws_caller_identity.current.account_id}", "arn:aws:s3:::rag-api-tfstate-${data.aws_caller_identity.current.account_id}/*"]
  }
}

resource "aws_iam_role_policy" "github_actions" {
  name   = "${var.project}-github-actions"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.github_actions.json
}
