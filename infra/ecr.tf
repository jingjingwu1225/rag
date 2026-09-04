resource "aws_ecr_repository" "api" {
  name                 = var.project
  image_tag_mutability = "MUTABLE"

  # Catches known CVEs in the base image on every push, at no cost.
  image_scanning_configuration {
    scan_on_push = true
  }
}

# Images are tagged with the git SHA, so without expiry this repository grows
# by ~1.2 GB per deploy forever.
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the 10 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}
