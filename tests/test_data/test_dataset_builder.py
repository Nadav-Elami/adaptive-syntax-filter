"""Tests for dataset builder functionality."""

import pytest
import numpy as np
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.adaptive_syntax_filter.data.dataset_builder import (
    DatasetMetadata,
    Dataset,
    analyze_sequence_statistics,
    validate_dataset_quality,
    DatasetBuilder,
    create_research_datasets
)
from src.adaptive_syntax_filter.data.sequence_generator import GenerationConfig


class TestDatasetMetadata:
    """Test suite for DatasetMetadata dataclass."""
    
    def test_dataset_metadata_creation(self):
        """Test basic DatasetMetadata creation."""
        metadata = DatasetMetadata(
            name="test_dataset",
            creation_time="2023-01-01T00:00:00",
            config={"test": "value"},
            alphabet_info={"alphabet": ["A", "B", "C"]},
            sequence_stats={"n_sequences": 100},
            validation_results={"validation_rate": 0.95}
        )
        
        assert metadata.name == "test_dataset"
        assert metadata.creation_time == "2023-01-01T00:00:00"
        assert metadata.config == {"test": "value"}
        assert metadata.alphabet_info == {"alphabet": ["A", "B", "C"]}
        assert metadata.sequence_stats == {"n_sequences": 100}
        assert metadata.validation_results == {"validation_rate": 0.95}
    
    def test_dataset_metadata_serialization(self):
        """Test that DatasetMetadata can be serialized."""
        metadata = DatasetMetadata(
            name="serialization_test",
            creation_time="2023-12-01T12:00:00",
            config={"param": 42},
            alphabet_info={"size": 10},
            sequence_stats={"count": 200},
            validation_results={"passed": True}
        )
        
        # Should be convertible to dict
        from dataclasses import asdict
        metadata_dict = asdict(metadata)
        
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["name"] == "serialization_test"
        assert metadata_dict["config"]["param"] == 42


class TestDataset:
    """Test suite for Dataset dataclass."""
    
    def test_dataset_creation(self):
        """Test basic Dataset creation."""
        sequences = [["A", "B"], ["B", "A", "C"]]
        trajectory = np.array([[1.0, 2.0], [3.0, 4.0]])
        metadata = DatasetMetadata(
            name="test",
            creation_time="2023-01-01T00:00:00",
            config={},
            alphabet_info={},
            sequence_stats={},
            validation_results={}
        )
        
        dataset = Dataset(
            sequences=sequences,
            parameter_trajectory=trajectory,
            metadata=metadata
        )
        
        assert dataset.sequences == sequences
        assert np.array_equal(dataset.parameter_trajectory, trajectory)
        assert dataset.metadata == metadata
    
    def test_dataset_with_empty_sequences(self):
        """Test Dataset creation with empty sequences."""
        dataset = Dataset(
            sequences=[],
            parameter_trajectory=np.array([]),
            metadata=DatasetMetadata(
                name="empty",
                creation_time="2023-01-01T00:00:00",
                config={},
                alphabet_info={},
                sequence_stats={},
                validation_results={}
            )
        )
        
        assert dataset.sequences == []
        assert len(dataset.parameter_trajectory) == 0
        assert dataset.metadata.name == "empty"


class TestAnalyzeSequenceStatistics:
    """Test suite for analyze_sequence_statistics function."""
    
    def test_analyze_basic_statistics(self):
        """Test basic sequence statistics analysis."""
        sequences = [
            ["START", "A", "B", "END"],
            ["START", "B", "A", "A", "END"],
            ["START", "A", "END"]
        ]
        alphabet = ["START", "A", "B", "C", "END"]
        
        stats = analyze_sequence_statistics(sequences, alphabet)
        
        # Check basic structure
        assert "n_sequences" in stats
        assert "length_stats" in stats
        assert "symbol_usage" in stats
        assert "phrase_usage" in stats
        assert "transition_stats" in stats
        assert "diversity_metrics" in stats
        
        # Check values
        assert stats["n_sequences"] == 3
        assert stats["length_stats"]["min"] == 3
        assert stats["length_stats"]["max"] == 5
        assert stats["length_stats"]["mean"] == (4 + 5 + 3) / 3
    
    def test_analyze_empty_sequences(self):
        """Test analysis with empty sequence list."""
        sequences = []
        alphabet = ["A", "B", "C"]
        
        stats = analyze_sequence_statistics(sequences, alphabet)
        
        assert "error" in stats
        assert stats["error"] == "No sequences to analyze"
    
    def test_analyze_symbol_frequencies(self):
        """Test symbol frequency calculation."""
        sequences = [
            ["A", "B", "A"],
            ["B", "B", "A"]
        ]
        alphabet = ["A", "B", "C"]
        
        stats = analyze_sequence_statistics(sequences, alphabet)
        
        # A appears 3 times, B appears 3 times, C appears 0 times
        expected_frequencies = {"A": 3/6, "B": 3/6, "C": 0.0}
        
        assert stats["symbol_usage"]["frequencies"] == expected_frequencies
        assert stats["symbol_usage"]["total_symbols"] == 6
    
    def test_analyze_transition_patterns(self):
        """Test transition pattern analysis."""
        sequences = [
            ["A", "B", "A"],
            ["A", "B", "B"]
        ]
        alphabet = ["A", "B"]
        
        stats = analyze_sequence_statistics(sequences, alphabet)
        
        # Should detect A->B (2 times), B->A (1 time), B->B (1 time)
        assert stats["transition_stats"]["total_transitions"] == 3
        assert ("A", "B") in [t[0] for t in stats["transition_stats"]["most_common"]]
    
    def test_analyze_diversity_metrics(self):
        """Test diversity metrics calculation."""
        sequences = [
            ["A", "B"],
            ["A", "B"],  # Duplicate
            ["B", "A"]
        ]
        alphabet = ["A", "B"]
        
        stats = analyze_sequence_statistics(sequences, alphabet)
        
        # 2 unique sequences out of 3 total
        assert stats["diversity_metrics"]["unique_sequences"] == 2
        assert stats["diversity_metrics"]["repetition_rate"] == 1 - 2/3
    
    def test_analyze_phrase_usage(self):
        """Test phrase usage analysis (excluding START/END)."""
        sequences = [
            ["START", "A", "B", "END"],
            ["START", "B", "A", "END"]
        ]
        alphabet = ["START", "A", "B", "C", "END"]
        
        stats = analyze_sequence_statistics(sequences, alphabet)
        
        # Phrase symbols are A, B, C (excluding START/END)
        phrase_usage = stats["phrase_usage"]
        assert "A" in phrase_usage["counts"]
        assert "B" in phrase_usage["counts"]
        assert "C" in phrase_usage["counts"]
        assert phrase_usage["counts"]["A"] == 2
        assert phrase_usage["counts"]["B"] == 2
        assert phrase_usage["counts"]["C"] == 0


class TestValidateDatasetQuality:
    """Test suite for validate_dataset_quality function."""
    
    def create_test_dataset(self, validation_rate=0.95, n_sequences=100, unique_rate=0.9):
        """Helper to create test dataset."""
        sequences = [["A", "B"] for _ in range(n_sequences)]
        if unique_rate < 1.0:
            # Add some duplicates
            n_duplicates = int(n_sequences * (1 - unique_rate))
            sequences.extend([["A", "B"] for _ in range(n_duplicates)])
        
        metadata = DatasetMetadata(
            name="test",
            creation_time="2023-01-01T00:00:00",
            config={},
            alphabet_info={"alphabet": ["A", "B"]},
            sequence_stats={
                "n_sequences": len(sequences),
                "length_stats": {"mean": 2.0, "std": 0.0, "min": 2, "max": 2}
            },
            validation_results={"validation_rate": validation_rate}
        )
        
        return Dataset(
            sequences=sequences,
            parameter_trajectory=np.array([]),
            metadata=metadata
        )
    
    def test_validate_high_quality_dataset(self):
        """Test validation of high-quality dataset."""
        dataset = self.create_test_dataset(validation_rate=0.98, unique_rate=0.95)
        
        results = validate_dataset_quality(dataset)
        
        assert "overall_quality" in results
        assert "passed_checks" in results
        assert "total_checks" in results
        assert "issues" in results
        assert "warnings" in results
        assert isinstance(results["issues"], list)
        assert isinstance(results["warnings"], list)
    
    def test_validate_with_custom_thresholds(self):
        """Test validation with custom quality thresholds."""
        dataset = self.create_test_dataset(validation_rate=0.90, unique_rate=0.85)
        
        custom_thresholds = {
            'min_validation_rate': 0.85,
            'min_unique_sequences': 0.80,
            'max_repetition_rate': 0.25,
            'min_symbol_diversity': 0.05,
            'max_length_cv': 3.0
        }
        
        results = validate_dataset_quality(dataset, custom_thresholds)
        
        # Should pass with relaxed thresholds
        assert "passed_checks" in results
        assert results["passed_checks"] > 0
    
    def test_validate_low_quality_dataset(self):
        """Test validation of low-quality dataset."""
        dataset = self.create_test_dataset(validation_rate=0.80, unique_rate=0.60)
        
        results = validate_dataset_quality(dataset)
        
        # Should have issues
        assert len(results["issues"]) > 0
        assert "Low validation rate" in str(results["issues"])
    
    def test_validate_error_handling(self):
        """Test validation error handling."""
        # Create dataset with malformed metadata
        sequences = [["A", "B"]]
        metadata = DatasetMetadata(
            name="malformed",
            creation_time="2023-01-01T00:00:00",
            config={},
            alphabet_info={},
            sequence_stats={},
            validation_results={}  # Missing required fields
        )
        
        dataset = Dataset(sequences=sequences, parameter_trajectory=np.array([]), metadata=metadata)
        
        # Should handle missing fields gracefully
        results = validate_dataset_quality(dataset)
        assert "overall_quality" in results


class TestDatasetBuilder:
    """Test suite for DatasetBuilder class."""
    
    def test_dataset_builder_initialization(self):
        """Test DatasetBuilder initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            assert builder.workspace_dir == Path(tmpdir)
            assert hasattr(builder, 'preset_configs')
            assert hasattr(builder, 'preset_alphabets')
            assert hasattr(builder, 'evolution_examples')
            assert builder.created_datasets == []
    
    def test_dataset_builder_default_workspace(self):
        """Test DatasetBuilder with default workspace."""
        builder = DatasetBuilder()
        
        # Should create default workspace
        assert builder.workspace_dir.exists()
        assert builder.workspace_dir.is_dir()
    
    @patch('src.adaptive_syntax_filter.data.dataset_builder.generate_dataset')
    @patch('src.adaptive_syntax_filter.data.dataset_builder.validate_generated_sequences')
    def test_create_dataset_basic(self, mock_validate, mock_generate):
        """Test basic dataset creation."""
        # Mock dependencies
        mock_sequences = [["START", "A", "B", "END"], ["START", "B", "A", "END"]]
        mock_trajectory = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_generate.return_value = (mock_sequences, mock_trajectory)
        mock_validate.return_value = {"validation_rate": 0.95, "errors": []}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            config = GenerationConfig(
                alphabet=['<', 'A', 'B', 'C', '>'],
                order=1,
                n_sequences=2,
                max_length=5
            )
            
            dataset = builder.create_dataset(config, name="test_dataset")
            
            # Verify dataset structure
            assert isinstance(dataset, Dataset)
            assert dataset.sequences == mock_sequences
            assert np.array_equal(dataset.parameter_trajectory, mock_trajectory)
            assert dataset.metadata.name == "test_dataset"
            assert dataset.metadata.validation_results["validation_rate"] == 0.95
            
            # Should be tracked
            assert len(builder.created_datasets) == 1
            assert builder.created_datasets[0] == "test_dataset"
    
    def test_create_from_preset(self):
        """Test dataset creation from preset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            # Mock the preset creation to avoid complex dependencies
            with patch.object(builder, 'create_dataset') as mock_create:
                mock_dataset = MagicMock(spec=Dataset)
                mock_create.return_value = mock_dataset
                
                dataset = builder.create_from_preset("bengalese_finch", name="preset_test")
                
                assert dataset == mock_dataset
                mock_create.assert_called_once()
    
    def test_create_from_preset_with_modifications(self):
        """Test preset creation with modifications."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            modifications = {"n_sequences": 50, "alphabet_size": 8}
            
            with patch.object(builder, 'create_dataset') as mock_create:
                mock_dataset = MagicMock(spec=Dataset)
                mock_create.return_value = mock_dataset
                
                dataset = builder.create_from_preset(
                    "canary", 
                    name="modified_preset",
                    modifications=modifications
                )
                
                assert dataset == mock_dataset
                mock_create.assert_called_once()
    
    def test_create_from_invalid_preset(self):
        """Test error handling for invalid preset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            with pytest.raises(ValueError, match="Unknown preset"):
                builder.create_from_preset("invalid_preset")
    
    def test_create_batch_datasets(self):
        """Test batch dataset creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            base_config = GenerationConfig(
                alphabet=['<', 'A', 'B', 'C', '>'],
                order=1,
                n_sequences=10,
                max_length=5
            )
            
            variations = [
                {"n_sequences": 20},
                {"alphabet": ['<', 'A', 'B', 'C', 'D', 'E', 'F', '>']},
                {"order": 2}
            ]
            
            with patch.object(builder, 'create_dataset') as mock_create:
                mock_datasets = [MagicMock(spec=Dataset) for _ in range(3)]
                mock_create.side_effect = mock_datasets
                
                datasets = builder.create_batch_datasets(
                    base_config, 
                    variations, 
                    name_prefix="batch"
                )
                
                assert len(datasets) == 3
                assert all(isinstance(d, MagicMock) for d in datasets)
                assert mock_create.call_count == 3
    
    def test_save_dataset_pickle(self):
        """Test saving dataset in pickle format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            # Create test dataset
            sequences = [["A", "B"], ["B", "A"]]
            metadata = DatasetMetadata(
                name="pickle_test",
                creation_time="2023-01-01T00:00:00",
                config={},
                alphabet_info={},
                sequence_stats={},
                validation_results={}
            )
            dataset = Dataset(sequences, np.array([]), metadata)
            
            filepath = builder.save_dataset(dataset, format='pickle')
            
            # Verify file exists
            assert filepath.exists()
            assert filepath.suffix == '.pickle'
            
            # Verify can be loaded
            with open(filepath, 'rb') as f:
                loaded_dataset = pickle.load(f)
            
            assert loaded_dataset.sequences == sequences
            assert loaded_dataset.metadata.name == "pickle_test"
    
    def test_save_dataset_json(self):
        """Test saving dataset in JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            sequences = [["A", "B"], ["B", "A"]]
            trajectory = np.array([[1.0, 2.0], [3.0, 4.0]])
            metadata = DatasetMetadata(
                name="json_test",
                creation_time="2023-01-01T00:00:00",
                config={"test": "value"},
                alphabet_info={"alphabet": ["A", "B"]},
                sequence_stats={"n_sequences": 2},
                validation_results={"validation_rate": 0.95}
            )
            dataset = Dataset(sequences, trajectory, metadata)
            
            filepath = builder.save_dataset(dataset, format='json', include_parameters=True)
            
            # Verify file exists and is valid JSON
            assert filepath.exists()
            assert filepath.suffix == '.json'
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert data["sequences"] == sequences
            assert data["parameter_trajectory"] == trajectory.tolist()
            assert data["metadata"]["name"] == "json_test"
    
    def test_save_dataset_txt(self):
        """Test saving dataset in text format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            sequences = [["A", "B"], ["B", "A"]]
            metadata = DatasetMetadata(
                name="txt_test",
                creation_time="2023-01-01T00:00:00",
                config={},
                alphabet_info={"alphabet": ["A", "B"]},
                sequence_stats={},
                validation_results={}
            )
            dataset = Dataset(sequences, np.array([]), metadata)
            
            filepath = builder.save_dataset(dataset, format='txt')
            
            # Verify file exists and has expected content
            assert filepath.exists()
            assert filepath.suffix == '.txt'
            
            content = filepath.read_text()
            assert "txt_test" in content
            assert "A B" in content
            assert "B A" in content
    
    def test_save_dataset_invalid_format(self):
        """Test error handling for invalid save format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            dataset = Dataset([], np.array([]), DatasetMetadata(
                name="test", creation_time="", config={}, 
                alphabet_info={}, sequence_stats={}, validation_results={}
            ))
            
            with pytest.raises(ValueError, match="Unknown format"):
                builder.save_dataset(dataset, format='invalid')
    
    def test_load_dataset_pickle(self):
        """Test loading dataset from pickle file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            # Create and save dataset
            original_dataset = Dataset(
                sequences=[["A", "B"]],
                parameter_trajectory=np.array([[1.0, 2.0]]),
                metadata=DatasetMetadata(
                    name="load_test", creation_time="", config={},
                    alphabet_info={}, sequence_stats={}, validation_results={}
                )
            )
            
            filepath = builder.save_dataset(original_dataset, format='pickle')
            
            # Load dataset
            loaded_dataset = builder.load_dataset(filepath)
            
            assert loaded_dataset.sequences == original_dataset.sequences
            assert np.array_equal(loaded_dataset.parameter_trajectory, original_dataset.parameter_trajectory)
            assert loaded_dataset.metadata.name == original_dataset.metadata.name
    
    def test_load_dataset_json(self):
        """Test loading dataset from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            # Create and save dataset
            original_dataset = Dataset(
                sequences=[["A", "B"]],
                parameter_trajectory=np.array([[1.0, 2.0]]),
                metadata=DatasetMetadata(
                    name="json_load_test", creation_time="", config={},
                    alphabet_info={}, sequence_stats={}, validation_results={}
                )
            )
            
            filepath = builder.save_dataset(original_dataset, format='json')
            
            # Load dataset
            loaded_dataset = builder.load_dataset(filepath)
            
            assert loaded_dataset.sequences == original_dataset.sequences
            assert np.array_equal(loaded_dataset.parameter_trajectory, original_dataset.parameter_trajectory)
            assert loaded_dataset.metadata.name == original_dataset.metadata.name
    
    def test_load_dataset_invalid_format(self):
        """Test error handling for invalid load format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            # Create file with unsupported extension
            invalid_file = Path(tmpdir) / "test.xml"
            invalid_file.write_text("invalid content")
            
            with pytest.raises(ValueError, match="Cannot load format"):
                builder.load_dataset(invalid_file)
    
    def test_list_presets(self):
        """Test listing available presets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            presets = builder.list_presets()
            
            assert "configs" in presets
            assert "alphabets" in presets
            assert "evolution_models" in presets
            assert isinstance(presets["configs"], dict)
            assert isinstance(presets["alphabets"], dict)
            assert isinstance(presets["evolution_models"], dict)
    
    def test_benchmark_performance(self):
        """Test performance benchmarking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            config = GenerationConfig(
                alphabet=['<', 'A', 'B', 'C', '>'],
                order=1,
                n_sequences=10,
                max_length=5
            )
            
            with patch('src.adaptive_syntax_filter.data.dataset_builder.generate_dataset') as mock_generate:
                # Mock generation to return consistent results
                mock_sequences = [["A", "B"] for _ in range(10)]
                mock_trajectory = np.array([[1.0, 2.0] for _ in range(5)])
                mock_generate.return_value = (mock_sequences, mock_trajectory)
                
                benchmark = builder.benchmark_performance(config, n_trials=2)
                
                assert "config" in benchmark
                assert "timing" in benchmark
                assert "memory" in benchmark
                assert "throughput" in benchmark
                
                # Check timing stats
                timing = benchmark["timing"]
                assert "mean_seconds" in timing
                assert "std_seconds" in timing
                assert "min_seconds" in timing
                assert "max_seconds" in timing
                
                # Check memory stats
                memory = benchmark["memory"]
                assert "mean_mb" in memory
                assert "trajectory_mb" in memory
                
                # Check throughput stats
                throughput = benchmark["throughput"]
                assert "sequences_per_second" in throughput
                assert "symbols_per_second" in throughput
    
    def test_get_workspace_summary(self):
        """Test workspace summary retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            summary = builder.get_workspace_summary()
            
            assert "workspace_dir" in summary
            assert "created_datasets" in summary
            assert "available_presets" in summary
            assert "available_alphabets" in summary
            
            assert summary["workspace_dir"] == str(tmpdir)
            assert summary["created_datasets"] == []
            assert isinstance(summary["available_presets"], list)
            assert isinstance(summary["available_alphabets"], list)


class TestCreateResearchDatasets:
    """Test suite for create_research_datasets function."""
    
    @patch('src.adaptive_syntax_filter.data.dataset_builder.DatasetBuilder')
    def test_create_research_datasets_basic(self, mock_builder_class):
        """Test basic research dataset creation."""
        # Mock the builder and its methods
        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        
        mock_datasets = {
            'bengalese_finch': MagicMock(spec=Dataset),
            'canary': MagicMock(spec=Dataset),
            'minimal': MagicMock(spec=Dataset)
        }
        
        mock_builder.create_from_preset.side_effect = lambda name: mock_datasets[name]
        mock_builder.save_dataset.return_value = Path("/fake/path")
        
        # Mock dataset metadata for printing
        for dataset_name, dataset in mock_datasets.items():
            dataset.metadata.sequence_stats = {'n_sequences': 100}
            dataset.metadata.validation_results = {'validation_rate': 0.95}
        
        with patch('builtins.print'):  # Suppress output
            datasets = create_research_datasets()
        
        # Verify all datasets created
        assert len(datasets) == 3
        assert 'bengalese_finch' in datasets
        assert 'canary' in datasets
        assert 'minimal' in datasets
        
        # Verify builder methods called
        assert mock_builder.create_from_preset.call_count == 3
        assert mock_builder.save_dataset.call_count == 3
    
    @patch('src.adaptive_syntax_filter.data.dataset_builder.DatasetBuilder')
    def test_create_research_datasets_with_output_dir(self, mock_builder_class):
        """Test research dataset creation with custom output directory."""
        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        
        mock_builder.create_from_preset.return_value = MagicMock(spec=Dataset)
        mock_builder.save_dataset.return_value = Path("/fake/path")
        
        # Mock metadata
        mock_builder.create_from_preset.return_value.metadata.sequence_stats = {'n_sequences': 50}
        mock_builder.create_from_preset.return_value.metadata.validation_results = {'validation_rate': 0.90}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('builtins.print'):  # Suppress output
                datasets = create_research_datasets(output_dir=tmpdir)
            
            # Verify builder initialized with custom directory
            mock_builder_class.assert_called_once_with(tmpdir)
            
            assert len(datasets) == 3


class TestIntegrationScenarios:
    """Integration tests for dataset builder functionality."""
    
    def test_full_dataset_creation_workflow(self):
        """Test complete dataset creation and save/load workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            # Create simple mock config
            config = GenerationConfig(
                alphabet=['<', 'A', '>'],
                order=1,
                n_sequences=5,
                max_length=4
            )
            
            # Mock the generation process
            with patch('src.adaptive_syntax_filter.data.dataset_builder.generate_dataset') as mock_generate:
                with patch('src.adaptive_syntax_filter.data.dataset_builder.validate_generated_sequences') as mock_validate:
                    
                    # Setup mocks
                    mock_sequences = [["A", "B"], ["B", "A"], ["A", "A"]]
                    mock_trajectory = np.array([[1.0, 2.0], [3.0, 4.0]])
                    mock_generate.return_value = (mock_sequences, mock_trajectory)
                    mock_validate.return_value = {"validation_rate": 0.95, "errors": []}
                    
                    # Create dataset
                    dataset = builder.create_dataset(config, name="integration_test")
                    
                    # Save in multiple formats
                    pickle_path = builder.save_dataset(dataset, format='pickle')
                    json_path = builder.save_dataset(dataset, format='json')
                    txt_path = builder.save_dataset(dataset, format='txt')
                    
                    # Verify files exist
                    assert pickle_path.exists()
                    assert json_path.exists()
                    assert txt_path.exists()
                    
                    # Load and verify
                    loaded_pickle = builder.load_dataset(pickle_path)
                    loaded_json = builder.load_dataset(json_path)
                    
                    assert loaded_pickle.sequences == mock_sequences
                    assert loaded_json.sequences == mock_sequences
                    assert np.array_equal(loaded_pickle.parameter_trajectory, mock_trajectory)
                    assert np.array_equal(loaded_json.parameter_trajectory, mock_trajectory)
    
    def test_batch_creation_and_validation(self):
        """Test batch dataset creation with quality validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = DatasetBuilder(tmpdir)
            
            base_config = GenerationConfig(
                alphabet=['<', 'A', 'B', '>'],
                order=1,
                n_sequences=10,
                max_length=5
            )
            
            variations = [
                {"n_sequences": 15},
                {"alphabet": ['<', 'A', 'B', 'C', '>']}
            ]
            
            with patch('src.adaptive_syntax_filter.data.dataset_builder.generate_dataset') as mock_generate:
                with patch('src.adaptive_syntax_filter.data.dataset_builder.validate_generated_sequences') as mock_validate:
                    
                    # Setup mocks
                    mock_sequences = [["A", "B", "C"] for _ in range(10)]
                    mock_trajectory = np.array([[1.0, 2.0, 3.0] for _ in range(5)])
                    mock_generate.return_value = (mock_sequences, mock_trajectory)
                    mock_validate.return_value = {"validation_rate": 0.98, "errors": []}
                    
                    # Create batch
                    datasets = builder.create_batch_datasets(
                        base_config, 
                        variations, 
                        name_prefix="batch"
                    )
                    
                    # Verify batch creation
                    assert len(datasets) == 2
                    
                    # Validate all datasets
                    for dataset in datasets:
                        quality_results = validate_dataset_quality(dataset)
                        assert "overall_quality" in quality_results 