import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from dataclasses import dataclass
import json
import logging
import warnings
from sklearn.metrics import mean_absolute_error, r2_score
from itertools import combinations, product
import math
from datetime import datetime

warnings.filterwarnings("ignore")


@dataclass
class ConditionalRule:
		"""Class to store conditional rules with metrics"""

		conditions: List[Dict[str, str]]
		formula: str
		metrics: Dict[str, float]

		def __str__(self):
				conditions_str = " AND ".join(
						[f"{cond['column']} == {cond['value']}" for cond in self.conditions]
				)
				return f"IF {conditions_str}: {self.formula}"


class RuleSight:
		"""
		RuleSight: A tool for extracting business rules from input-output data
		"""

		def __init__(self, logging_level=logging.INFO, error_tolerance=1e-6):
				"""Initialize RuleSight with configuration parameters"""
				self.logger = self._setup_logging(logging_level)
				self.numeric_rules = {}
				self.categorical_rules = {}
				self.error_tolerance = error_tolerance

		def _setup_logging(self, level) -> logging.Logger:
				"""Configure logging system"""
				logger = logging.getLogger("RuleSight")
				logger.setLevel(level)
				if not logger.handlers:
						handler = logging.StreamHandler()
						formatter = logging.Formatter(
								"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
						)
						handler.setFormatter(formatter)
						logger.addHandler(handler)
				return logger

		def _is_valid_numeric(self, x: pd.Series) -> bool:
				"""Validate numeric series for computation"""
				try:
						if x is None or x.empty:
								return False
						return not (
								x.isin([np.inf, -np.inf]).any()
								or x.isna().any()
								or (np.abs(x) > 1e308).any()
						)
				except Exception as e:
						self.logger.debug(f"Numeric validation failed: {str(e)}")
						return False

		def _safe_numeric_operation(
				self, x: pd.Series, y: pd.Series, operation: str
		) -> Optional[pd.Series]:
				"""Safely perform numeric operations with validation"""
				try:
						if not (self._is_valid_numeric(x) and self._is_valid_numeric(y)):
								return None

						result = None
						if operation == "+":
								result = x + y
						elif operation == "*":
								if (
										np.log(np.abs(x.replace(0, 1))) + np.log(np.abs(y.replace(0, 1)))
										< 700
								).all():
										result = x * y
						elif operation == "-":
								result = x - y

						return result if self._is_valid_numeric(result) else None
				except Exception as e:
						self.logger.debug(f"Operation failed: {str(e)}")
						return None

		def _detect_complex_numeric_patterns(
				self, X: pd.DataFrame, y: pd.Series
		) -> List[ConditionalRule]:
				"""Detect complex numeric patterns with multiple conditions"""
				rules = []
				operations = ["+", "*", "-"]
				categorical_cols = X.select_dtypes(include=["object"]).columns
				numeric_cols = X.select_dtypes(include=[np.number]).columns

				if numeric_cols.empty or categorical_cols.empty:
						return rules

				for cat_col in categorical_cols:
						for cat_val in X[cat_col].unique():
								primary_condition = {"column": cat_col, "value": cat_val}
								mask_primary = X[cat_col] == cat_val

								# Try single condition patterns
								if mask_primary.sum() >= 2:
										X_subset = X[mask_primary]
										y_subset = y[mask_primary]
										single_pattern = self._find_best_pattern(
												X_subset[numeric_cols], y_subset, operations
										)
										if single_pattern:
												rules.append(
														ConditionalRule(
																conditions=[primary_condition],
																formula=single_pattern["formula"],
																metrics=single_pattern["metrics"],
														)
												)

								# Try patterns with two conditions
								for second_cat_col in categorical_cols:
										if second_cat_col != cat_col:
												for second_cat_val in X[second_cat_col].unique():
														conditions = [
																primary_condition,
																{"column": second_cat_col, "value": second_cat_val},
														]
														mask = mask_primary & (X[second_cat_col] == second_cat_val)

														if mask.sum() >= 2:
																X_subset = X[mask]
																y_subset = y[mask]
																pattern = self._find_best_pattern(
																		X_subset[numeric_cols], y_subset, operations
																)
																if pattern:
																		rules.append(
																				ConditionalRule(
																						conditions=conditions,
																						formula=pattern["formula"],
																						metrics=pattern["metrics"],
																				)
																		)

				return rules

		def _find_best_pattern(
				self, X: pd.DataFrame, y: pd.Series, operations: List[str]
		) -> Optional[Dict]:
				"""Find the best matching pattern for a subset of data"""
				best_error = float("inf")
				best_pattern = None

				for col1, col2 in combinations(X.columns, 2):
						for op in operations:
								result = self._safe_numeric_operation(X[col1], X[col2], op)
								if result is not None:
										error = mean_absolute_error(y, result)
										relative_error = error / (y.mean() or 1)

										if (
												relative_error < self.error_tolerance
												and relative_error < best_error
										):
												best_error = relative_error
												best_pattern = {
														"formula": f"{col1} {op} {col2}",
														"metrics": {
																"mae": error,
																"relative_error": relative_error,
																"r2": r2_score(y, result),
														},
												}

				return best_pattern

		def _detect_string_combination(self, X: pd.DataFrame, y: pd.Series) -> Dict:
				"""Detect string combination patterns"""
				categorical_cols = X.select_dtypes(include=["object"]).columns
				if categorical_cols.empty:
						return {}

				# First letter combinations
				for col1 in categorical_cols:
						if all(y == X[col1].str[0]):
								return {
										"type": "first_letter",
										"columns": [col1],
										"operation": "first_letter",
								}

						for col2 in categorical_cols:
								if col1 != col2:
										combined = X[col1].str[0] + X[col2].str[0]
										if all(combined == y):
												return {
														"type": "string_combination",
														"columns": [col1, col2],
														"operation": "first_letters_concat",
												}

				return {}

		def extract_rules(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
				"""Extract rules from input and output data"""
				try:
						self.logger.info("Starting rule extraction process")

						# Validate input
						if input_df.empty or output_df.empty:
								raise ValueError("Input or output DataFrame is empty")
						if len(input_df) != len(output_df):
								raise ValueError("Input and output DataFrames must have same length")
						if input_df.isna().any().any() or output_df.isna().any().any():
								raise ValueError("Data contains missing values")

						# Process each output column
						for col in output_df.columns:
								if pd.api.types.is_numeric_dtype(output_df[col]):
										rules = self._detect_complex_numeric_patterns(
												input_df, output_df[col]
										)
										if rules:
												self.numeric_rules[col] = rules
								else:
										rules = self._detect_string_combination(input_df, output_df[col])
										if rules:
												self.categorical_rules[col] = rules

				except Exception as e:
						self.logger.error(f"Rule extraction failed: {str(e)}")
						raise

		def get_rules_summary(self) -> Dict:
				"""Generate human-readable summary of discovered rules"""
				summary = {"numeric_rules": {}, "categorical_rules": {}}

				# Format numeric rules
				for col, rules in self.numeric_rules.items():
						summary["numeric_rules"][col] = []
						for rule in rules:
								conditions_str = " AND ".join(
										[f"{cond['column']} == {cond['value']}" for cond in rule.conditions]
								)
								summary["numeric_rules"][col].append(
										f"IF {conditions_str}: {rule.formula}"
								)

				# Format categorical rules
				for col, rules in self.categorical_rules.items():
						rule_type = rules.get("type", "")
						if rule_type == "string_combination":
								col1, col2 = rules["columns"]
								summary["categorical_rules"][
										col
								] = f"Combine first letters of {col1} and {col2}"
						elif rule_type == "first_letter":
								col_name = rules["columns"][0]
								summary["categorical_rules"][col] = f"First letter of {col_name}"

				return summary

		def export_results(self, file_path: str, format: str = "json"):
				"""
				Export rules to a file

				Args:
												file_path: Path to save the results
												format: Export format ('json' or 'txt')
				"""
				try:
						results = {
								"rules": self.get_rules_summary(),
								"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
								"configuration": {"error_tolerance": self.error_tolerance},
						}

						with open(file_path, "w") as f:
								if format == "json":
										json.dump(results, f, indent=2)
								elif format == "txt":
										# Write header
										f.write("=== RuleSight Analysis Results ===\n\n")
										f.write(f"Generated on: {results['timestamp']}\n")
										f.write(
												f"Error Tolerance: {results['configuration']['error_tolerance']}\n\n"
										)

										# Write numeric rules
										f.write("=== Numeric Rules ===\n")
										for col, rules in results["rules"]["numeric_rules"].items():
												f.write(f"\nColumn: {col}\n")
												for rule in rules:
														f.write(f"  {rule}\n")

										# Write categorical rules
										f.write("\n=== Categorical Rules ===\n")
										for col, rule in results["rules"]["categorical_rules"].items():
												f.write(f"\nColumn: {col}\n")
												f.write(f"  {rule}\n")
								else:
										raise ValueError(f"Unsupported format: {format}")

				except Exception as e:
						self.logger.error(f"Export failed: {str(e)}")
						raise


# ================ Data Generation Utilities ================
def generate_simple_sample_data(
		n_samples: int = 100000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
		"""Generate simple sample data for testing"""
		np.random.seed(42)

		input_data = {
				"Amount": np.random.uniform(1000, 50000, n_samples),
				"ProductType": np.random.choice(["GOLD", "PLAT", "SILV"], n_samples),
				"Score": np.random.uniform(300, 850, n_samples),
				"Region": np.random.choice(["NORTH", "SOUTH", "EAST", "WEST"], n_samples),
				"Commission": np.random.uniform(100000, 500000, n_samples),
				"CommissionType": np.random.choice(["Y", "N"], n_samples),
		}

		input_df = pd.DataFrame(input_data)

		output_data = {
			'RiskScore': np.where(
					input_df['ProductType'] == 'GOLD',
					input_df['Amount'] * input_df['Score'] + 20000,
					np.where(
							((input_df['ProductType'] == 'PLAT') & (input_df['CommissionType'] == 'Y')),
							input_df['Amount'] + input_df['Score'] + input_df['Commission'],
							np.where(
									((input_df['ProductType'] == 'PLAT') & (input_df['CommissionType'] == 'N')),
									input_df['Amount'] + input_df['Score'] - input_df['Commission'],
									input_df['Amount'] * input_df['Score']  # default case for SILV
							)
					)
		),
		'Type': input_df['ProductType'].str[0]
}

		output_df = pd.DataFrame(output_data)

		# Export to CSV files
		input_df.to_csv("input_data.csv", index=False)
		output_df.to_csv("output_data.csv", index=False)
		return input_df, output_df


# ================ Main Execution ================


def main():
		# Create simple test data
		input_data, output_data = generate_simple_sample_data()

		# Initialize RuleSight with relaxed error tolerance
		rulesight = RuleSight(error_tolerance=1e-4)

		# Extract rules
		rulesight.extract_rules(input_data, output_data)

		# Display results
		rulesight.export_results("rules_output.txt", "txt")

		# Export results
		rulesight.export_results("rules_output.json")

		print(input_data)
		print(output_data)


if __name__ == "__main__":
		main()
