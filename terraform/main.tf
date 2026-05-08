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

# SECURITY GROUP FOR EC2
resource "aws_security_group" "retail_sg" {
  name        = "retail-ops-sg"
  description = "Allow inbound traffic for Retail Ops app"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 INSTANCE FOR DOCKER DEPLOYMENT
resource "aws_instance" "retail_app_server" {
  ami           = "ami-0c7217cdde317cfec" # Ubuntu 22.04 LTS (us-east-1)
  instance_type = "t3.micro"
  key_name      = "retail-ops-key" # Assumes you have this key pair in AWS

  vpc_security_group_ids = [aws_security_group.retail_sg.id]

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
  }

  user_data = <<-EOF
              #!/bin/bash
              sudo apt-get update
              sudo apt-get install -y docker.io docker-compose git
              sudo systemctl start docker
              sudo systemctl enable docker
              sudo usermod -aG docker ubuntu
              EOF

  tags = {
    Name        = "Retail-Ops-App-Server"
    Environment = "Production"
    Project     = "Retail-Ops-Forecasting"
  }
}

output "public_ip" {
  value = aws_instance.retail_app_server.public_ip
}
