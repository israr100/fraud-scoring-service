variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "app_name" {
  type    = string
  default = "fraud-scoring-service"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "database_url" {
  type        = string
  description = "SQLAlchemy URL. Use an RDS Postgres endpoint in production."
  default     = "sqlite:////app/predictions.db"
}
