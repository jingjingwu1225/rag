output "ecr_repository_url" {
  description = "Push target for the API image."
  value       = aws_ecr_repository.api.repository_url
}

output "github_actions_role_arn" {
  description = "Set as AWS_ROLE_ARN in the GitHub Actions workflow."
  value       = aws_iam_role.github_actions.arn
}

output "history_table_name" {
  description = "DynamoDB table holding conversation history."
  value       = aws_dynamodb_table.history.name
}

output "openai_secret_arn" {
  description = "Secrets Manager ARN for the OpenAI key (value set out-of-band)."
  value       = aws_secretsmanager_secret.openai.arn
}

output "api_key_secret_arn" {
  description = "Secrets Manager ARN for the client API key (value set out-of-band)."
  value       = aws_secretsmanager_secret.api_key.arn
}

output "apprunner_instance_role_arn" {
  description = "Role the running container assumes."
  value       = aws_iam_role.apprunner_instance.arn
}

output "apprunner_access_role_arn" {
  description = "Role App Runner uses to pull from ECR."
  value       = aws_iam_role.apprunner_access.arn
}
