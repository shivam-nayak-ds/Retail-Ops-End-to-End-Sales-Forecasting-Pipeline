# PROVIDER CONFIGURATION
provider "aws" {
  region = "us-east-1"
}

# S3 BUCKET FOR MODEL ARTIFACTS (DVC REMOTE)
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "retail-ops-model-artifacts-001"
  acl    = "private"

  tags = {
    Name        = "Model Artifact Storage"
    Environment = "Production"
    Project     = "Retail-Ops-Forecasting"
  }
}

# EKS CLUSTER SKELETON (FOR K8S DEPLOYMENT)
# resource "aws_eks_cluster" "retail_cluster" {
#   name     = "retail-ops-cluster"
#   role_arn = aws_iam_role.eks_role.arn
#   ...
# }
