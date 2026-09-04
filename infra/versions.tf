terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state, not local. The moment CI runs `terraform apply`, a local
  # state file is worthless — the runner has none, so it would plan against an
  # empty state and try to recreate every resource that already exists.
  #
  # use_lockfile is native S3 locking (Terraform >= 1.10). The old
  # DynamoDB lock table is deprecated and no longer needed.
  #
  # The bucket itself is bootstrapped out-of-band (see infra/README.md) —
  # Terraform cannot create the bucket that holds its own state.
  backend "s3" {
    bucket       = "rag-api-tfstate-227536679105"
    key          = "rag-api/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "rag-api"
      ManagedBy = "terraform"
    }
  }
}
