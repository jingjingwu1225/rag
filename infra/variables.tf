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
