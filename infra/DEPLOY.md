# Deploying to AWS

The image is deploy-ready; these are the steps to run it on **your** AWS account.
You need the AWS CLI, Docker, and Terraform installed and `aws configure` done.

## 1. Provision infrastructure (ECR + ECS Fargate)

```bash
cd infra/terraform
terraform init
terraform apply            # creates ECR repo, ECS cluster, task def, service
```

Note the `ecr_repository_url` output.

## 2. Build, tag, and push the image

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
REPO=$(terraform -chdir=infra/terraform output -raw ecr_repository_url)

python -m src.train                       # bake a fresh model artifact into the image
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker build -f docker/Dockerfile -t $REPO:latest .
docker push $REPO:latest
```

## 3. Roll the service

```bash
aws ecs update-service --cluster fraud-scoring-service-cluster \
  --service fraud-scoring-service --force-new-deployment
```

Find the task's public IP in the ECS console (or via `aws ecs describe-tasks`) and hit
`http://<public-ip>:8000/health`.

## Production hardening (next steps, called out honestly)

- **Put it behind an Application Load Balancer** with an ACM TLS cert; drop the
  public-IP-on-task pattern and restrict the security group to the ALB.
- **Swap SQLite for RDS Postgres** — set `database_url` to the RDS endpoint
  (`postgresql+psycopg://...`) in `terraform.tfvars`. The app already supports it.
- **Store the model in S3 / a model registry** and load at startup, so you deploy
  new models without rebuilding the image.
- **Autoscale** the ECS service on CPU / request count.

## CI → ECR (optional)

Add `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` as GitHub repo secrets, then add a
job to `.github/workflows/ci.yml`:

```yaml
  push-image:
    runs-on: ubuntu-latest
    needs: docker
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - uses: aws-actions/amazon-ecr-login@v2
      - run: |
          pip install -r requirements.txt && python -m src.train
          docker build -f docker/Dockerfile -t $ECR_REPO:${{ github.sha }} .
          docker push $ECR_REPO:${{ github.sha }}
```
