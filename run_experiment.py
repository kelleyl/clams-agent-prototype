#!/usr/bin/env python3
"""
CLI for running CLAMS experiments with different LLM models.

Usage:
    # Run experiment with multiple models
    python run_experiment.py run ocr-comparison \
        --models "openai:gpt-4o-mini,anthropic:claude-3-haiku,ollama:llama3.2" \
        --task ocr

    # Compare experiment results
    python run_experiment.py compare \
        --experiments "exp1,exp2,exp3" \
        --metrics "accuracy,f1_score"

    # List experiments
    python run_experiment.py list --task ocr --limit 20

    # Show experiment details
    python run_experiment.py show <experiment_id>

    # Delete experiment
    python run_experiment.py delete <experiment_id>
"""
import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.experiment_tracker import (
    ExperimentTracker, ModelConfig, LLMProvider, ExperimentMetrics
)
from utils.model_providers import get_provider, Message
from utils.config import ConfigManager


def parse_model_string(model_str: str) -> ModelConfig:
    """
    Parse model string to ModelConfig.

    Formats supported:
    - "openai:gpt-4o-mini" -> OpenAI provider with gpt-4o-mini
    - "anthropic:claude-3-haiku" -> Anthropic provider
    - "ollama:llama3.2" -> Ollama provider
    - "llama3.2" -> Defaults to Ollama
    - "gpt-4o-mini" -> Defaults to OpenAI (if starts with gpt)
    - "claude-3" -> Defaults to Anthropic (if starts with claude)
    """
    model_str = model_str.strip()

    # Check for explicit provider:model format
    if ':' in model_str:
        parts = model_str.split(':', 1)
        provider_str = parts[0].lower()
        model_name = parts[1]

        if provider_str in ['openai', 'anthropic', 'ollama']:
            return ModelConfig(
                provider=LLMProvider(provider_str),
                model_name=model_name
            )

    # Infer provider from model name
    model_lower = model_str.lower()
    if model_lower.startswith('gpt') or model_lower.startswith('o1'):
        return ModelConfig(provider=LLMProvider.OPENAI, model_name=model_str)
    elif model_lower.startswith('claude'):
        return ModelConfig(provider=LLMProvider.ANTHROPIC, model_name=model_str)
    else:
        # Default to Ollama for local models
        return ModelConfig(provider=LLMProvider.OLLAMA, model_name=model_str)


def run_experiment_command(args):
    """Run an experiment with specified models."""
    config = ConfigManager().get_config()
    tracker = ExperimentTracker(
        config.experiment.experiments_dir,
        storage_backend=config.experiment.storage_backend
    )

    # Parse models
    models = [parse_model_string(m.strip()) for m in args.models.split(',')]

    print(f"\n{'='*60}")
    print(f"CLAMS Experiment: {args.name}")
    print(f"Task: {args.task or 'default'}")
    print(f"Models to test: {len(models)}")
    print(f"{'='*60}\n")

    results = []

    for model_config in models:
        print(f"\n--- Testing: {model_config} ---")

        try:
            provider = get_provider(model_config)

            if not provider.is_available():
                print(f"  SKIPPED: Provider not available")
                continue

            with tracker.start_experiment(
                name=args.name,
                model_config=model_config,
                task_type=args.task or "default",
                dataset_name=args.dataset or ""
            ) as exp:
                # Run a simple test prompt
                test_prompt = args.prompt or "What CLAMS tools would you recommend for extracting text from video?"

                print(f"  Prompt: {test_prompt[:50]}...")

                response = provider.generate(
                    prompt=test_prompt,
                    system_prompt="You are a helpful assistant for video analysis."
                )

                # Log the response
                exp.log_prediction(
                    input_data=test_prompt,
                    prediction=response.content
                )
                exp.log_tokens(response.prompt_tokens, response.completion_tokens)
                exp.log_metrics({
                    "latency_seconds": response.latency_seconds
                })

                print(f"  Response length: {len(response.content)} chars")
                print(f"  Tokens: {response.total_tokens}")
                print(f"  Latency: {response.latency_seconds:.2f}s")
                print(f"  Saved as: {exp.experiment_id}")

                results.append({
                    "model": str(model_config),
                    "experiment_id": exp.experiment_id,
                    "tokens": response.total_tokens,
                    "latency": response.latency_seconds
                })

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['model']}: {r['tokens']} tokens, {r['latency']:.2f}s")
        print(f"    ID: {r['experiment_id']}")
    print()


def compare_command(args):
    """Compare multiple experiments."""
    config = ConfigManager().get_config()
    tracker = ExperimentTracker(config.experiment.experiments_dir)

    experiment_ids = [e.strip() for e in args.experiments.split(',')]
    metrics = args.metrics.split(',') if args.metrics else None

    comparison = tracker.compare_experiments(experiment_ids, metrics)

    if "error" in comparison:
        print(f"Error: {comparison['error']}")
        return

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*60}\n")

    # Print experiment details
    for exp in comparison['experiments']:
        print(f"ID: {exp['id']}")
        print(f"  Model: {exp['provider']}:{exp['model']}")
        print(f"  Timestamp: {exp['timestamp']}")
        print(f"  Metrics:")
        for metric, value in exp['metrics'].items():
            if value is not None:
                print(f"    {metric}: {value}")
        print()

    # Print summary
    print("METRICS SUMMARY:")
    print("-" * 40)
    for metric, summary in comparison['metrics_summary'].items():
        direction = "lower" if summary['lower_is_better'] else "higher"
        print(f"\n{metric} ({direction} is better):")
        print(f"  Min: {summary['min']:.4f}")
        print(f"  Max: {summary['max']:.4f}")
        print(f"  Mean: {summary['mean']:.4f}")
        print(f"  Best: {summary['best_experiment']}")


def list_command(args):
    """List experiments with optional filters."""
    config = ConfigManager().get_config()
    tracker = ExperimentTracker(config.experiment.experiments_dir)

    experiments = tracker.list_experiments(
        name=args.name,
        model=args.model,
        task=args.task,
        limit=args.limit
    )

    if not experiments:
        print("No experiments found.")
        return

    print(f"\nFound {len(experiments)} experiments:\n")

    # Table header
    print(f"{'ID':<45} {'Model':<25} {'Task':<15} {'Timestamp':<20}")
    print("-" * 105)

    for exp in experiments:
        model_str = f"{exp.model_config.provider.value if hasattr(exp.model_config.provider, 'value') else exp.model_config.provider}:{exp.model_config.model_name}"
        print(f"{exp.experiment_id:<45} {model_str:<25} {exp.task_type:<15} {exp.timestamp[:19]:<20}")

    print()


def show_command(args):
    """Show details of a specific experiment."""
    config = ConfigManager().get_config()
    tracker = ExperimentTracker(config.experiment.experiments_dir)

    result = tracker.load_experiment(args.experiment_id)

    if not result:
        print(f"Experiment not found: {args.experiment_id}")
        return

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {result.experiment_id}")
    print(f"{'='*60}\n")

    print(f"Name: {result.experiment_name}")
    print(f"Task: {result.task_type}")
    print(f"Dataset: {result.dataset_name}")
    print(f"Timestamp: {result.timestamp}")
    print()

    print("MODEL CONFIGURATION:")
    print(f"  Provider: {result.model_config.provider}")
    print(f"  Model: {result.model_config.model_name}")
    print(f"  Temperature: {result.model_config.temperature}")
    print()

    print("METRICS:")
    metrics_dict = result.metrics.to_dict()
    for key, value in metrics_dict.items():
        if value is not None and value != 0 and key != 'custom':
            print(f"  {key}: {value}")
    if result.metrics.custom:
        for key, value in result.metrics.custom.items():
            print(f"  {key}: {value}")
    print()

    print("SYSTEM INFO:")
    for key, value in result.system_info.items():
        print(f"  {key}: {value}")
    print()

    if result.predictions:
        print(f"PREDICTIONS: {len(result.predictions)} logged")
        if args.verbose:
            for i, pred in enumerate(result.predictions[:5]):
                print(f"\n  [{i+1}]")
                print(f"    Input: {str(pred.get('input', ''))[:100]}...")
                print(f"    Output: {str(pred.get('prediction', ''))[:200]}...")
    print()

    if result.errors:
        print(f"ERRORS: {len(result.errors)}")
        for error in result.errors:
            print(f"  - {error}")
        print()


def delete_command(args):
    """Delete an experiment."""
    config = ConfigManager().get_config()
    tracker = ExperimentTracker(config.experiment.experiments_dir)

    if not args.force:
        confirm = input(f"Delete experiment {args.experiment_id}? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    if tracker.delete_experiment(args.experiment_id):
        print(f"Deleted: {args.experiment_id}")
    else:
        print(f"Experiment not found: {args.experiment_id}")


def models_command(args):
    """List available models for each provider."""
    print("\nAvailable LLM Providers and Models:\n")

    # OpenAI
    print("OPENAI:")
    print("  gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo")
    print("  Usage: openai:gpt-4o-mini")
    print()

    # Anthropic
    print("ANTHROPIC:")
    print("  claude-3-opus, claude-3-sonnet, claude-3-haiku")
    print("  claude-3-5-sonnet, claude-3-5-haiku")
    print("  Usage: anthropic:claude-3-haiku")
    print()

    # Ollama
    print("OLLAMA (local):")
    try:
        from utils.model_providers import OllamaProvider
        from utils.experiment_tracker import ModelConfig, LLMProvider

        config = ModelConfig(provider=LLMProvider.OLLAMA, model_name="")
        ollama = OllamaProvider(config)

        if ollama.is_available():
            models = ollama.list_models()
            if models:
                for model in models[:10]:
                    print(f"  {model}")
                if len(models) > 10:
                    print(f"  ... and {len(models) - 10} more")
            else:
                print("  No models found. Run: ollama pull llama3.2")
        else:
            print("  Ollama not running. Start with: ollama serve")
    except Exception as e:
        print(f"  Could not list Ollama models: {e}")

    print("\n  Usage: ollama:llama3.2 or just llama3.2")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="CLAMS Experiment Runner - Compare LLM models for pipeline tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s run my-test --models "gpt-4o-mini,claude-3-haiku"
  %(prog)s list --task ocr --limit 10
  %(prog)s compare --experiments "exp1,exp2"
  %(prog)s show <experiment_id>
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run experiment command
    run_parser = subparsers.add_parser('run', help='Run an experiment with models')
    run_parser.add_argument('name', help='Experiment name')
    run_parser.add_argument('--models', '-m', required=True,
                          help='Comma-separated models (e.g., "openai:gpt-4o-mini,ollama:llama3.2")')
    run_parser.add_argument('--task', '-t', help='Task type (e.g., ocr, transcription)')
    run_parser.add_argument('--dataset', '-d', help='Dataset name or path')
    run_parser.add_argument('--prompt', '-p', help='Test prompt to use')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('--experiments', '-e', required=True,
                               help='Comma-separated experiment IDs')
    compare_parser.add_argument('--metrics', help='Metrics to compare (comma-separated)')

    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--name', '-n', help='Filter by experiment name')
    list_parser.add_argument('--model', '-m', help='Filter by model name')
    list_parser.add_argument('--task', '-t', help='Filter by task type')
    list_parser.add_argument('--limit', '-l', type=int, default=20, help='Max results')

    # Show command
    show_parser = subparsers.add_parser('show', help='Show experiment details')
    show_parser.add_argument('experiment_id', help='Experiment ID')
    show_parser.add_argument('--verbose', '-v', action='store_true', help='Show predictions')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete an experiment')
    delete_parser.add_argument('experiment_id', help='Experiment ID')
    delete_parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')

    # Models command
    models_parser = subparsers.add_parser('models', help='List available models')

    args = parser.parse_args()

    if args.command == 'run':
        run_experiment_command(args)
    elif args.command == 'compare':
        compare_command(args)
    elif args.command == 'list':
        list_command(args)
    elif args.command == 'show':
        show_command(args)
    elif args.command == 'delete':
        delete_command(args)
    elif args.command == 'models':
        models_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
