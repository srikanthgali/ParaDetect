#!/usr/bin/env python3
"""
ParaDetect Project Structure Generator
Creates a complete MLOps pipeline structure for AI text detection
Modern components and pipelines architecture
"""

import os
import logging
from pathlib import Path
from typing import List

class ParaDetectProjectGenerator:
    """Generates complete project structure for ParaDetect"""

    def __init__(self, project_name: str = "para-detect"):
        self.project_name = project_name
        self.base_path = Path(".") 
        
    def create_project_structure(self):
        """Create complete project directory structure"""
        
        logging.info(f"Creating project structure for {self.project_name}")
        
        # Track files and directories created
        created_count = 0
        
        # Define directory structure
        directories = [
            ".github/workflows",
            "data/raw", "data/processed", "data/interim",
            "configs/model_configs", "configs/deployment",
            "src/para_detect/core", 
            "src/para_detect/components", 
            "src/para_detect/pipelines",
            "src/para_detect/utils",
            "scripts/aws",
            "tests/unit", "tests/integration", "tests/fixtures",
            "docs", 
            "artifacts/models", "artifacts/logs", "artifacts/metrics", "artifacts/reports",
            "notebooks", 
            "api/routes", "api/middleware", 
            "docker"
        ]
        
        # Create directories
        for directory in directories:
            dir_path = self.base_path / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created_count += 1
                logging.info(f"Created directory: {dir_path}")
            else:
                logging.info(f"Directory already exists: {dir_path}")
            
        logging.info("All directories created successfully.")

        # Create individual files        
        individual_files = [
            ".gitignore",
            "README.md",
            "requirements.txt",
            "setup.py",    
            "CHANGELOG.md",
            "LICENSE",
            "Makefile",
            ".github/workflows/ci-cd-pipeline.yml",
            ".github/workflows/model-training.yml",
            
            "configs/__init__.py",
            "configs/config.yaml",
            "configs/model_configs/deberta_config.yaml",
            "configs/model_configs/training_config.yaml",
            "configs/deployment/aws_config.yaml",
            "configs/deployment/local_config.yaml",
            "configs/deployment/colab_config.yaml",
            
            "docker/Dockerfile",
            "docker/docker-compose.yml",
            "docker/Dockerfile.training",
            "docker/Dockerfile.inference",
            
            "src/__init__.py",
            "src/para_detect/__init__.py",            
            "src/para_detect/core/__init__.py",
            "src/para_detect/core/base.py",
            "src/para_detect/core/config_manager.py",
            "src/para_detect/core/logger.py",
            "src/para_detect/core/exceptions.py",
            
            "src/para_detect/components/__init__.py",
            "src/para_detect/components/data_ingestion.py",
            "src/para_detect/components/data_preprocessing.py",
            "src/para_detect/components/data_validation.py",
            "src/para_detect/components/feature_engineering.py",
            "src/para_detect/components/model_training.py",
            "src/para_detect/components/model_evaluation.py",
            "src/para_detect/components/model_validation.py",
            "src/para_detect/components/model_registration.py",
            "src/para_detect/components/model_deployment.py",
            "src/para_detect/components/monitoring.py",
            
            "src/para_detect/pipelines/__init__.py",
            "src/para_detect/pipelines/training_pipeline.py",
            "src/para_detect/pipelines/inference_pipeline.py",
            "src/para_detect/pipelines/batch_prediction_pipeline.py",
            "src/para_detect/pipelines/deployment_pipeline.py",
            "src/para_detect/pipelines/retraining_pipeline.py",
            
            "src/para_detect/utils/__init__.py",
            "src/para_detect/utils/helpers.py",
            "src/para_detect/utils/validators.py",
            "src/para_detect/utils/model_utils.py",
            "src/para_detect/utils/data_utils.py",
            
            "docs/overview.md",
            "docs/architecture.md",
            "docs/components.md",
            "docs/pipelines.md",
            "docs/deployment.md",
            "docs/troubleshooting.md",            

            "notebooks/01_EDA_and_Data_Preparation.ipynb",
            "notebooks/02_Model_Training_and_Evaluation.ipynb",
            "notebooks/03_ParaDetect_Gradio_Demo.ipynb",
            "notebooks/04_Pipeline_Testing.ipynb",

            "api/__init__.py",
            "api/main.py",
            "api/routes/__init__.py",
            "api/routes/prediction.py",
            "api/routes/monitoring.py",
            "api/routes/health.py",
            "api/middleware/__init__.py",
            "api/middleware/auth.py",
            "api/middleware/logging.py",
            "api/middleware/rate_limiting.py",
            
            "scripts/setup_environment.py",
            "scripts/run_training_pipeline.py",
            "scripts/run_inference_pipeline.py",
            "scripts/deploy_model.py",
            "scripts/aws/create_sagemaker_endpoint.py",
            "scripts/aws/lambda_deploy.py",
            "scripts/aws/ecs_deploy.py",
            "scripts/aws/cleanup_resources.py",
            
            "tests/__init__.py",
            "tests/unit/test_components.py",
            "tests/unit/test_pipelines.py",
            "tests/unit/test_utils.py",
            "tests/integration/test_full_pipeline.py",
            "tests/integration/test_api_endpoints.py",
            "tests/fixtures/sample_data.json",
            "tests/fixtures/mock_responses.json"
        ]

        # Create individual files
        for individual_file in individual_files:
            file_path = self.base_path / individual_file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if not file_path.exists():
                file_path.touch()
                created_count += 1
                logging.info(f"Created file: {file_path}")
            else:
                logging.info(f"File already exists: {file_path}")

        # Create .gitkeep for empty directories
        gitkeep_dirs = [
            "artifacts/models",
            "artifacts/logs", 
            "artifacts/metrics",
            "artifacts/reports",
            "data/raw", 
            "data/processed", 
            "data/interim"
        ]
        
        for gitkeep_dir in gitkeep_dirs:
            gitkeep_path = self.base_path / gitkeep_dir / ".gitkeep"
            if not gitkeep_path.exists():
                gitkeep_path.touch()
                created_count += 1
                logging.info(f"Created .gitkeep: {gitkeep_path}")
            else:
                logging.info(f".gitkeep already exists: {gitkeep_path}")

        logging.info(f"ParaDetect project structure processed successfully at: {self.base_path}")
        logging.info(f"Total new files and directories created: {created_count}")
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    generator = ParaDetectProjectGenerator()
    generator.create_project_structure()    