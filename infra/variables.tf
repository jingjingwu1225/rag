variable "region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Name prefix for all resources."
  type        = string
  default     = "rag-api"
}

variable "github_repo" {
  description = "GitHub repo allowed to assume the CI role, as owner/name."
  type        = string
  default     = "jingjingwu1225/rag"
}

variable "github_repo_id" {
  description = <<-EOT
    Numeric owner and repo IDs, as "ownerID/repoID".

    GitHub's OIDC subject claim is not the documented
    "repo:OWNER/NAME:ref:..." — it embeds immutable numeric IDs:
    "repo:OWNER@ownerID/NAME@repoID:ref:...". Matching only the documented
    form silently never matches, and STS reports it as
    "Not authorized to perform sts:AssumeRoleWithWebIdentity", which looks
    like a permissions problem rather than a string mismatch.

    Read the real values from a workflow that prints its token claims, or:
      gh api repos/OWNER/NAME --jq '"\(.owner.id)/\(.id)"'
  EOT
  type        = string
  default     = "49615883/1357487784"
}

variable "budget_alert_email" {
  description = "Email address for budget threshold alerts."
  type        = string
  default     = "jenniew1225@gmail.com"
}

variable "monthly_budget_usd" {
  description = "Monthly spend budget. Alerts fire at 50% and 100% of this."
  type        = number
  default     = 30
}

variable "history_ttl_days" {
  description = "Days before a conversation's stored history expires."
  type        = number
  default     = 1
}
