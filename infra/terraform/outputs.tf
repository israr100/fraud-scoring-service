output "ecr_repository_url" {
  value       = aws_ecr_repository.app.repository_url
  description = "Push your image here."
}

output "ecs_cluster" {
  value = aws_ecs_cluster.app.name
}

output "log_group" {
  value = aws_cloudwatch_log_group.app.name
}
