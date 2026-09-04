# The OpenAI key lives here and nowhere else.
#
# Deliberately NO aws_secretsmanager_secret_version resource: writing the
# value through Terraform puts it in terraform.tfstate in plaintext, and that
# state file lives in S3. The secret is created empty here and populated
# out-of-band, once:
#
#   aws secretsmanager put-secret-value \
#     --secret-id rag-api/openai \
#     --secret-string 'sk-...'
#
# So the value exists in exactly two places: Secrets Manager, and the
# developer's local .env. Never in git, the image, or Terraform state.
resource "aws_secretsmanager_secret" "openai" {
  name        = "${var.project}/openai"
  description = "OpenAI API key, injected into the service at runtime."

  # Short window so a mistaken delete can be undone but doesn't linger.
  recovery_window_in_days = 7
}

# Shared secret required by the API's X-API-Key header. Same rule: created
# empty, value set out-of-band.
resource "aws_secretsmanager_secret" "api_key" {
  name        = "${var.project}/api-key"
  description = "Shared secret clients must send as X-API-Key."

  recovery_window_in_days = 7
}
