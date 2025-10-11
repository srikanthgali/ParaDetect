"""
Inference Pipeline for ParaDetect
Handles real-time and batch predictions with CLI and server modes
"""

import os
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from para_detect.core.config_manager import ConfigurationManager
from para_detect.entities.inference_config import InferenceConfig
from para_detect.core.exceptions import (
    ParaDetectException,
    ModelPredictionError,
    DeviceError,
)
from para_detect.constants import DEVICE_PRIORITY, REVERSE_LABEL_MAPPING
from para_detect import get_logger


class InferencePipeline:
    """
    Comprehensive inference pipeline for ParaDetect.

    Features:
    - Single text prediction with confidence scores
    - Batch processing for CSV/Parquet files
    - Efficient batching with device optimization
    - Preprocessing and post-processing
    - Performance monitoring and timing
    - Error handling with graceful degradation
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        model_path: Optional[str] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize inference pipeline.

        Args:
            config: Inference configuration
            model_path: Path to trained model
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.config = config or self._load_inference_config()
        self.logger = get_logger(self.__class__.__name__)

        # Override model path if provided
        if model_path:
            self.config = InferenceConfig(
                **{**self.config.__dict__, "model_path": model_path}
            )

        # Initialize device
        self.device = self._detect_device()
        self.logger.info(f"🔧 Using device: {self.device}")

        # Model components
        self.model = None
        self.tokenizer = None
        self.classifier = None

        # Performance tracking
        self.stats = {
            "total_predictions": 0,
            "total_inference_time": 0.0,
            "average_inference_time": 0.0,
            "errors": 0,
        }

        # Initialize pipeline
        self.is_initialized = False
        self._initialize_pipeline()

    def _load_inference_config(self) -> InferenceConfig:
        """Load inference configuration from config manager."""
        try:
            # Try to get inference config from config manager
            inference_config_dict = self.config_manager.base_config.get("inference", {})

            return InferenceConfig(
                batch_size=inference_config_dict.get("batch_size", 32),
                max_length=inference_config_dict.get("max_length", 512),
                device_preference=inference_config_dict.get("device_preference", None),
                text_column=inference_config_dict.get("text_column", "text"),
                preprocessing_enabled=inference_config_dict.get(
                    "preprocessing_enabled", True
                ),
                include_probabilities=inference_config_dict.get(
                    "include_probabilities", True
                ),
                include_confidence=inference_config_dict.get(
                    "include_confidence", True
                ),
                confidence_threshold=inference_config_dict.get(
                    "confidence_threshold", 0.5
                ),
                enable_monitoring=inference_config_dict.get("enable_monitoring", True),
                log_predictions=inference_config_dict.get("log_predictions", False),
                chunk_size=inference_config_dict.get("chunk_size", 1000),
                progress_bar=inference_config_dict.get("progress_bar", True),
                skip_errors=inference_config_dict.get("skip_errors", True),
                max_retries=inference_config_dict.get("max_retries", 3),
            )
        except Exception as e:
            self.logger.warning(
                f"Error loading inference config, using defaults: {str(e)}"
            )
            return InferenceConfig()

    def _detect_device(self) -> torch.device:
        """Detect optimal device for inference."""
        try:
            if self.config.device_preference:
                if self.config.device_preference == "auto":
                    pass  # Continue with auto-detection
                else:
                    device = torch.device(self.config.device_preference)
                    if device.type == "cuda" and not torch.cuda.is_available():
                        self.logger.warning(
                            "CUDA requested but not available, falling back to auto-detection"
                        )
                    elif device.type == "mps" and not torch.backends.mps.is_available():
                        self.logger.warning(
                            "MPS requested but not available, falling back to auto-detection"
                        )
                    else:
                        return device

            # Auto-detection
            for device_type in DEVICE_PRIORITY:
                if device_type == "cuda" and torch.cuda.is_available():
                    return torch.device("cuda")
                elif device_type == "mps" and torch.backends.mps.is_available():
                    return torch.device("mps")
                elif device_type == "cpu":
                    return torch.device("cpu")

            return torch.device("cpu")

        except Exception as e:
            raise DeviceError(f"Failed to detect device: {str(e)}") from e

    def _initialize_pipeline(self):
        """Initialize inference pipeline components."""
        try:
            self.logger.info("🚀 Initializing inference pipeline...")

            # Load model and tokenizer
            model_path = self._resolve_model_path()
            self.logger.info(f"📥 Loading model from: {model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,  # Enable fast tokenizer for better performance
            )

            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=(
                    torch.float16 if self.device.type == "cuda" else torch.float32
                ),
                device_map=None,  # We'll move manually for better control
            )

            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            # Create pipeline for batch processing
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True,
                truncation=True,
                max_length=self.config.max_length,
                batch_size=self.config.batch_size,
            )

            self.is_initialized = True
            self.logger.info("✅ Inference pipeline initialized successfully")

            # Log model info
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"📊 Model info: {num_params:,} parameters")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize inference pipeline: {str(e)}")
            raise ParaDetectException(
                f"Inference pipeline initialization failed: {str(e)}"
            ) from e

    def _resolve_model_path(self) -> str:
        """Resolve model path from config or find latest model."""
        if self.config.model_path:
            if Path(self.config.model_path).exists():
                return self.config.model_path
            else:
                raise ParaDetectException(
                    f"Model path does not exist: {self.config.model_path}"
                )

        # Try to find latest model in artifacts
        artifacts_dir = Path("artifacts/models")
        if artifacts_dir.exists():
            model_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
            if model_dirs:
                # Sort by modification time and get latest
                latest_model = max(model_dirs, key=lambda x: x.stat().st_mtime)
                model_path = latest_model / "final_model"
                if model_path.exists():
                    return str(model_path)

        # Try local registry
        registry_dir = Path("artifacts/model_registry")
        if registry_dir.exists():
            index_file = registry_dir / "registry_index.json"
            if index_file.exists():
                with open(index_file, "r") as f:
                    index = json.load(f)
                if index.get("latest"):
                    latest_version = index["latest"]
                    model_path = registry_dir / f"v_{latest_version}" / "model"
                    if model_path.exists():
                        return str(model_path)

        raise ParaDetectException(
            "No trained model found. Please run training pipeline first."
        )

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Make prediction on single text sample.

        Args:
            text: Input text to classify

        Returns:
            Dict containing prediction results with confidence and probabilities
        """
        if not self.is_initialized:
            raise ParaDetectException("Pipeline not initialized")

        try:
            start_time = time.time()

            # Preprocess text if enabled
            processed_text = (
                self._preprocess_text(text)
                if self.config.preprocessing_enabled
                else text
            )

            # Tokenize and move to device
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1)

            # Extract results
            pred_label = prediction.item()
            pred_probs = probabilities.cpu().numpy()[0]
            confidence = float(pred_probs.max())

            inference_time = time.time() - start_time

            # Prepare result
            result = {
                "prediction": pred_label,
                "prediction_label": REVERSE_LABEL_MAPPING.get(
                    pred_label, f"class_{pred_label}"
                ),
                "confidence": confidence,
                "inference_time": inference_time,
                "original_text_length": len(text),
                "processed_text_length": len(processed_text),
                "error": False,
            }

            # Add probabilities if requested
            if self.config.include_probabilities:
                result["probabilities"] = {
                    REVERSE_LABEL_MAPPING.get(i, f"class_{i}"): float(prob)
                    for i, prob in enumerate(pred_probs)
                }

            # Update stats
            self._update_stats(inference_time, success=True)

            # Log prediction if enabled
            if self.config.log_predictions:
                self._log_prediction(text, result)

            return result

        except Exception as e:
            self.logger.error(f"Single prediction failed: {str(e)}")
            self._update_stats(0, success=False)

            if self.config.skip_errors:
                return {
                    "error": True,
                    "error_message": str(e),
                    "original_text": text[:100] + "..." if len(text) > 100 else text,
                }
            else:
                raise ModelPredictionError(f"Single prediction failed: {str(e)}") from e

    def predict_batch(
        self, texts: List[str], show_progress: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on batch of texts with efficient processing.

        Args:
            texts: List of input texts to classify
            show_progress: Show progress bar (overrides config)

        Returns:
            List of prediction results
        """
        if not self.is_initialized:
            raise ParaDetectException("Pipeline not initialized")

        if not texts:
            return []

        try:
            show_progress = (
                show_progress if show_progress is not None else self.config.progress_bar
            )
            self.logger.info(f"🔄 Processing batch of {len(texts)} texts...")

            results = []
            batch_size = self.config.batch_size

            # Process in batches
            if show_progress:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(
                        range(0, len(texts), batch_size), desc="Processing batches"
                    )
                except ImportError:
                    self.logger.warning("tqdm not available, progress bar disabled")
                    iterator = range(0, len(texts), batch_size)
            else:
                iterator = range(0, len(texts), batch_size)

            for i in iterator:
                batch_texts = texts[i : i + batch_size]

                try:
                    # Preprocess batch if enabled
                    if self.config.preprocessing_enabled:
                        processed_batch = [
                            self._preprocess_text(text) for text in batch_texts
                        ]
                    else:
                        processed_batch = batch_texts

                    # Use pipeline for batch processing
                    start_time = time.time()
                    batch_predictions = self.classifier(processed_batch)
                    batch_time = time.time() - start_time

                    # Process results
                    for j, (original_text, pred_result) in enumerate(
                        zip(batch_texts, batch_predictions)
                    ):
                        try:
                            # Extract prediction info
                            if isinstance(pred_result, list):
                                # Multiple scores returned
                                pred_scores = {
                                    item["label"]: item["score"] for item in pred_result
                                }

                                # Find prediction with highest score
                                best_pred = max(pred_result, key=lambda x: x["score"])
                                pred_label = self._label_to_int(best_pred["label"])
                                confidence = best_pred["score"]
                            else:
                                # Single prediction
                                pred_label = self._label_to_int(pred_result["label"])
                                confidence = pred_result["score"]
                                pred_scores = {pred_result["label"]: confidence}

                            # Prepare result
                            result = {
                                "prediction": pred_label,
                                "prediction_label": REVERSE_LABEL_MAPPING.get(
                                    pred_label, f"class_{pred_label}"
                                ),
                                "confidence": confidence,
                                "inference_time": batch_time
                                / len(batch_texts),  # Approximate per-text time
                                "original_text_length": len(original_text),
                                "processed_text_length": len(
                                    processed_batch[j]
                                    if self.config.preprocessing_enabled
                                    else original_text
                                ),
                                "error": False,
                            }

                            # Add probabilities if requested
                            if self.config.include_probabilities:
                                result["probabilities"] = pred_scores

                            results.append(result)
                            self._update_stats(result["inference_time"], success=True)

                        except Exception as e:
                            self.logger.warning(
                                f"Failed to process text {i+j}: {str(e)}"
                            )
                            error_result = {
                                "error": True,
                                "error_message": str(e),
                                "original_text": (
                                    original_text[:100] + "..."
                                    if len(original_text) > 100
                                    else original_text
                                ),
                            }
                            results.append(error_result)
                            self._update_stats(0, success=False)

                except Exception as e:
                    self.logger.error(
                        f"Batch processing failed for batch starting at {i}: {str(e)}"
                    )
                    # Add error results for entire batch
                    for text in batch_texts:
                        error_result = {
                            "error": True,
                            "error_message": str(e),
                            "original_text": (
                                text[:100] + "..." if len(text) > 100 else text
                            ),
                        }
                        results.append(error_result)
                        self._update_stats(0, success=False)

            self.logger.info(f"✅ Batch processing completed: {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"Batch prediction failed: {str(e)}")
            raise ModelPredictionError(f"Batch prediction failed: {str(e)}") from e

    def predict_from_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        text_column: str = None,
        chunk_size: Optional[int] = None,
    ) -> str:
        """
        Make predictions on data from CSV/Parquet file.

        Args:
            input_file: Path to input file (CSV or Parquet)
            output_file: Path to output file (optional)
            text_column: Name of text column (optional)
            chunk_size: Processing chunk size (optional)

        Returns:
            str: Path to output file with predictions
        """
        try:
            input_path = Path(input_file)
            if not input_path.exists():
                raise ParaDetectException(f"Input file does not exist: {input_file}")

            self.logger.info(f"📁 Processing file: {input_file}")

            # Determine output file
            if not output_file:
                output_file = (
                    input_path.parent
                    / f"{input_path.stem}_predictions{input_path.suffix}"
                )

            # Determine text column
            text_column = text_column or self.config.text_column

            # Determine chunk size
            chunk_size = chunk_size or self.config.chunk_size

            # Load data
            if input_path.suffix.lower() == ".csv":
                df = pd.read_csv(input_file)
            elif input_path.suffix.lower() in [".parquet", ".pq"]:
                df = pd.read_parquet(input_file)
            else:
                raise ParaDetectException(
                    f"Unsupported file format: {input_path.suffix}"
                )

            if text_column not in df.columns:
                raise ParaDetectException(f"Column '{text_column}' not found in file")

            self.logger.info(f"📊 Processing {len(df)} rows...")

            # Process in chunks for large files
            all_predictions = []

            for chunk_start in range(0, len(df), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(df))
                chunk_texts = df[text_column].iloc[chunk_start:chunk_end].tolist()

                self.logger.info(
                    f"Processing chunk {chunk_start//chunk_size + 1}/{(len(df) + chunk_size - 1)//chunk_size}"
                )

                # Get predictions for chunk
                chunk_predictions = self.predict_batch(chunk_texts)
                all_predictions.extend(chunk_predictions)

            # Add predictions to dataframe
            df["prediction"] = [p.get("prediction", None) for p in all_predictions]
            df["prediction_label"] = [
                p.get("prediction_label", None) for p in all_predictions
            ]
            df["confidence"] = [p.get("confidence", None) for p in all_predictions]
            df["inference_time"] = [
                p.get("inference_time", None) for p in all_predictions
            ]
            df["processing_error"] = [p.get("error", False) for p in all_predictions]

            # Add probabilities if included
            if self.config.include_probabilities and all_predictions:
                if "probabilities" in all_predictions[0]:
                    prob_keys = list(all_predictions[0]["probabilities"].keys())
                    for key in prob_keys:
                        df[f"prob_{key}"] = [
                            p.get("probabilities", {}).get(key, None)
                            for p in all_predictions
                        ]

            # Save results
            if output_file.endswith(".csv"):
                df.to_csv(output_file, index=False)
            elif output_file.endswith((".parquet", ".pq")):
                df.to_parquet(output_file, index=False)
            else:
                # Default to CSV
                df.to_csv(output_file, index=False)

            self.logger.info(f"✅ Predictions saved to: {output_file}")

            # Log summary statistics
            if all_predictions:
                successful_preds = [
                    p for p in all_predictions if not p.get("error", False)
                ]
                error_count = len(all_predictions) - len(successful_preds)

                self.logger.info(f"📈 Summary:")
                self.logger.info(f"   Total predictions: {len(all_predictions)}")
                self.logger.info(f"   Successful: {len(successful_preds)}")
                self.logger.info(f"   Errors: {error_count}")

                if successful_preds:
                    avg_confidence = np.mean(
                        [p["confidence"] for p in successful_preds]
                    )
                    avg_time = np.mean([p["inference_time"] for p in successful_preds])
                    self.logger.info(f"   Avg confidence: {avg_confidence:.4f}")
                    self.logger.info(f"   Avg inference time: {avg_time:.4f}s")

            return str(output_file)

        except Exception as e:
            self.logger.error(f"File prediction failed: {str(e)}")
            raise ModelPredictionError(f"File prediction failed: {str(e)}") from e

    def _preprocess_text(self, text: str) -> str:
        """Apply basic text preprocessing."""
        if not isinstance(text, str):
            text = str(text)

        # Basic cleaning
        text = text.strip()

        # Remove extra whitespace
        import re

        text = re.sub(r"\s+", " ", text)

        return text

    def _label_to_int(self, label: str) -> int:
        """Convert string label to integer."""
        if label.upper() in ["LABEL_0", "0", "HUMAN"]:
            return 0
        elif label.upper() in ["LABEL_1", "1", "AI"]:
            return 1
        else:
            # Try to extract number from label
            import re

            match = re.search(r"\d+", str(label))
            if match:
                return int(match.group())
            return 0  # Default to 0

    def _update_stats(self, inference_time: float, success: bool = True):
        """Update performance statistics."""
        self.stats["total_predictions"] += 1
        if success:
            self.stats["total_inference_time"] += inference_time
            self.stats["average_inference_time"] = self.stats[
                "total_inference_time"
            ] / (self.stats["total_predictions"] - self.stats["errors"])
        else:
            self.stats["errors"] += 1

    def _log_prediction(self, text: str, result: Dict[str, Any]):
        """Log prediction for monitoring."""
        if self.config.enable_monitoring:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "text_length": len(text),
                "prediction": result.get("prediction"),
                "confidence": result.get("confidence"),
                "inference_time": result.get("inference_time"),
            }
            self.logger.debug(f"Prediction logged: {log_entry}")

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.stats.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        if not self.is_initialized:
            return {"error": "Pipeline not initialized"}

        try:
            num_params = sum(p.numel() for p in self.model.parameters())

            return {
                "model_path": self.config.model_path,
                "model_type": self.model.__class__.__name__,
                "num_parameters": num_params,
                "device": str(self.device),
                "max_length": self.config.max_length,
                "batch_size": self.config.batch_size,
                "tokenizer_type": self.tokenizer.__class__.__name__,
                "vocab_size": len(self.tokenizer),
            }
        except Exception as e:
            return {"error": f"Failed to get model info: {str(e)}"}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on inference pipeline."""
        try:
            # Test with sample text
            test_result = self.predict_single("This is a test text for health check.")

            return {
                "status": "healthy",
                "model_loaded": self.is_initialized,
                "test_prediction_successful": not test_result.get("error", False),
                "model_info": self.get_model_info(),
                "performance_stats": self.get_stats(),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": self.is_initialized,
                "timestamp": datetime.now().isoformat(),
            }


def main():
    """CLI interface for inference pipeline."""
    parser = argparse.ArgumentParser(description="ParaDetect Inference Pipeline")

    # Model configuration
    parser.add_argument(
        "--model-path", type=str, help="Path to trained model directory"
    )

    parser.add_argument("--config", type=str, help="Path to custom config file")

    # Prediction modes
    parser.add_argument("--text", type=str, help="Single text to predict")

    parser.add_argument(
        "--file", type=str, help="Input file (CSV/Parquet) for batch prediction"
    )

    parser.add_argument("--output", type=str, help="Output file for batch predictions")

    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of text column in input file",
    )

    # Processing options
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for processing"
    )

    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum sequence length"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to use for inference",
    )

    # Utility options
    parser.add_argument(
        "--health-check", action="store_true", help="Perform health check and exit"
    )

    parser.add_argument(
        "--model-info", action="store_true", help="Show model information and exit"
    )

    parser.add_argument(
        "--server",
        action="store_true",
        help="Start inference server (requires FastAPI)",
    )

    parser.add_argument(
        "--port", type=int, default=8000, help="Port for inference server"
    )

    args = parser.parse_args()

    try:
        # Initialize configuration
        if args.config:
            config_manager = ConfigurationManager(args.config)
        else:
            config_manager = ConfigurationManager()

        # Create inference config
        inference_config = InferenceConfig(
            model_path=args.model_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device_preference=args.device,
            text_column=args.text_column,
        )

        # Initialize pipeline
        pipeline = InferencePipeline(
            config=inference_config, config_manager=config_manager
        )

        # Handle different modes
        if args.health_check:
            health = pipeline.health_check()
            print("🏥 Health Check Results:")
            print(f"  Status: {health['status']}")
            print(f"  Model Loaded: {health['model_loaded']}")
            if health["status"] == "healthy":
                print(f"  Test Prediction: ✅")
            else:
                print(f"  Error: {health.get('error', 'Unknown')}")
            return

        if args.model_info:
            info = pipeline.get_model_info()
            print("🤖 Model Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            return

        if args.server:
            # Start FastAPI server
            try:
                from fastapi import FastAPI, HTTPException
                from pydantic import BaseModel
                import uvicorn

                app = FastAPI(title="ParaDetect Inference API")

                class PredictionRequest(BaseModel):
                    text: str

                class BatchPredictionRequest(BaseModel):
                    texts: List[str]

                @app.get("/health")
                def health_check():
                    return pipeline.health_check()

                @app.get("/model-info")
                def model_info():
                    return pipeline.get_model_info()

                @app.post("/predict")
                def predict(request: PredictionRequest):
                    try:
                        return pipeline.predict_single(request.text)
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=str(e))

                @app.post("/predict-batch")
                def predict_batch(request: BatchPredictionRequest):
                    try:
                        return pipeline.predict_batch(request.texts)
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=str(e))

                print(f"🚀 Starting inference server on port {args.port}")
                print(f"📍 API docs available at: http://localhost:{args.port}/docs")
                uvicorn.run(app, host="0.0.0.0", port=args.port)

            except ImportError:
                print(
                    "❌ FastAPI not available. Install with: pip install fastapi uvicorn"
                )
                return

        if args.text:
            # Single prediction
            print(f"🔍 Predicting single text...")
            result = pipeline.predict_single(args.text)

            print(f"📊 Results:")
            print(f"  Text: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
            print(f"  Prediction: {result['prediction_label']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Inference Time: {result['inference_time']:.4f}s")

            if "probabilities" in result:
                print(f"  Probabilities:")
                for label, prob in result["probabilities"].items():
                    print(f"    {label}: {prob:.4f}")

        elif args.file:
            # Batch prediction
            print(f"📁 Processing file: {args.file}")
            output_file = pipeline.predict_from_file(
                input_file=args.file,
                output_file=args.output,
                text_column=args.text_column,
            )
            print(f"✅ Results saved to: {output_file}")

            # Show stats
            stats = pipeline.get_stats()
            print(f"📈 Performance Stats:")
            print(f"  Total Predictions: {stats['total_predictions']}")
            print(f"  Average Time: {stats['average_inference_time']:.4f}s")
            print(f"  Errors: {stats['errors']}")

        else:
            print(
                "❌ No prediction mode specified. Use --text, --file, --health-check, --model-info, or --server"
            )
            parser.print_help()

    except Exception as e:
        print(f"❌ Pipeline execution failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
