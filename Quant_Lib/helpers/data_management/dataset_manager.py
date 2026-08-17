"""
Database management for dynamic datasets in Quant_Lib.

This module provides SQLite-based data management for market data that is updated
from Refinitiv workspace Excel files. It handles:
- Excel to SQLite data ingestion
- Data versioning and history tracking
- Incremental updates
- Data validation and integrity checks
- Export back to CSV for package inclusion
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Tuple
import hashlib
import logging


class DatasetManager:
    """Manage market datasets with SQLite backend and Excel interface.
    
    This class provides a robust interface for managing time-series market data
    that needs regular updates from external sources (Refinitiv Excel).
    
    Features:
    - SQLite database for efficient storage and querying
    - Data versioning and audit trails
    - Incremental updates with conflict resolution
    - Data validation and integrity checks
    - Excel import/export capabilities
    - Package-ready CSV generation
    """
    
    def __init__(self, db_path: str = "datasets/market_data.db"):
        """Initialize the dataset manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Main data table with time-series structure
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    value REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source_file TEXT,
                    UNIQUE(dataset_name, date, column_name)
                )
            """)
            
            # Metadata table for dataset information
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    dataset_name TEXT PRIMARY KEY,
                    description TEXT,
                    source_type TEXT,
                    update_frequency TEXT,
                    last_updated TIMESTAMP,
                    record_count INTEGER,
                    checksum TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Audit log for tracking changes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    records_affected INTEGER,
                    source_file TEXT,
                    user_notes TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_dataset_date ON market_data(dataset_name, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_dataset ON audit_log(dataset_name)")
    
    def import_from_excel(self, 
                         excel_path: str,
                         sheet_mappings: Dict[str, str],
                         update_mode: str = "upsert") -> Dict[str, int]:
        """Import data from Refinitiv Excel workspace.
        
        Args:
            excel_path: Path to Excel file
            sheet_mappings: Dict mapping sheet names to dataset names
                          e.g., {"Sheet1": "Histocopy_SOFR", "Gold": "HISTOCOPY_Gold"}
            update_mode: "upsert" (default), "replace", or "append"
            
        Returns:
            Dict with dataset names and number of records imported
        """
        results = {}
        excel_path = Path(excel_path)
        
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        try:
            with pd.ExcelFile(excel_path) as xls:
                for sheet_name, dataset_name in sheet_mappings.items():
                    if sheet_name not in xls.sheet_names:
                        self.logger.warning(f"Sheet {sheet_name} not found in {excel_path}")
                        continue
                    
                    # Read data from Excel
                    df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0)
                    df.index = pd.to_datetime(df.index)  # Ensure datetime index
                    
                    # Import to database
                    records_imported = self._import_dataframe(
                        df, dataset_name, str(excel_path), update_mode
                    )
                    results[dataset_name] = records_imported
                    
                    self.logger.info(f"Imported {records_imported} records for {dataset_name}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error importing from Excel: {e}")
            raise
    
    def _import_dataframe(self,
                         df: pd.DataFrame,
                         dataset_name: str,
                         source_file: str,
                         update_mode: str) -> int:
        """Import a DataFrame into the database.
        
        Args:
            df: DataFrame with datetime index and numeric columns
            dataset_name: Name of the dataset
            source_file: Path to source file
            update_mode: Import mode
            
        Returns:
            Number of records imported
        """
        records_imported = 0
        
        with sqlite3.connect(self.db_path) as conn:
            if update_mode == "replace":
                # Delete existing data for this dataset
                conn.execute("DELETE FROM market_data WHERE dataset_name = ?", (dataset_name,))
            
            # Prepare data for insertion
            insert_data = []
            for date_idx, row in df.iterrows():
                date_str = date_idx.strftime('%Y-%m-%d')
                for column_name, value in row.items():
                    if pd.notna(value):  # Skip NaN values
                        insert_data.append((
                            dataset_name,
                            date_str,
                            str(column_name),
                            float(value),
                            source_file
                        ))
            
            if update_mode == "upsert":
                # Use INSERT OR REPLACE for upsert behavior
                conn.executemany("""
                    INSERT OR REPLACE INTO market_data 
                    (dataset_name, date, column_name, value, source_file, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, insert_data)
            else:  # append
                conn.executemany("""
                    INSERT OR IGNORE INTO market_data 
                    (dataset_name, date, column_name, value, source_file)
                    VALUES (?, ?, ?, ?, ?)
                """, insert_data)
            
            records_imported = len(insert_data)
            
            # Update metadata
            self._update_metadata(conn, dataset_name, df, source_file)
            
            # Log the operation
            conn.execute("""
                INSERT INTO audit_log (dataset_name, operation, records_affected, source_file)
                VALUES (?, ?, ?, ?)
            """, (dataset_name, f"import_{update_mode}", records_imported, source_file))
        
        return records_imported
    
    def _update_metadata(self,
                        conn: sqlite3.Connection,
                        dataset_name: str,
                        df: pd.DataFrame,
                        source_file: str):
        """Update dataset metadata."""
        # Calculate checksum for data integrity
        checksum = hashlib.md5(df.to_string().encode()).hexdigest()
        record_count = len(df)
        
        conn.execute("""
            INSERT OR REPLACE INTO dataset_metadata 
            (dataset_name, source_type, last_updated, record_count, checksum)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?)
        """, (dataset_name, "excel", record_count, checksum))
    
    def export_to_csv(self,
                     dataset_name: str,
                     output_path: Optional[str] = None,
                     date_range: Optional[Tuple[str, str]] = None) -> str:
        """Export dataset to CSV for package inclusion.
        
        Args:
            dataset_name: Name of dataset to export
            output_path: Output file path (optional)
            date_range: Tuple of (start_date, end_date) strings (optional)
            
        Returns:
            Path to exported CSV file
        """
        if output_path is None:
            # Special case for gold
            if dataset_name.lower() == "histocopy_gold":
                output_path = "datasets/HISTOCOPY_Gold.csv"
            else:
                output_path = f"datasets/{dataset_name}.csv"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        # Build query
        query = """
            SELECT date, column_name, value
            FROM market_data
            WHERE dataset_name = ?
        """
        params = [dataset_name]
        
        if date_range:
            query += " AND date BETWEEN ? AND ?"
            params.extend(date_range)
        
        query += " ORDER BY date, column_name"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            raise ValueError(f"No data found for dataset: {dataset_name}")
        
        # Pivot to create time-series format
        pivot_df = df.pivot(index='date', columns='column_name', values='value')
        pivot_df.index = pd.to_datetime(pivot_df.index)
        pivot_df = pivot_df.sort_index(ascending=False)  # Latest first
        
        # Export to CSV
        pivot_df.to_csv(output_path)
        self.logger.info(f"Exported {len(pivot_df)} records to {output_path}")
        
        return str(output_path)
    
    def get_dataset_info(self, dataset_name: Optional[str] = None) -> pd.DataFrame:
        """Get information about datasets.
        
        Args:
            dataset_name: Specific dataset name (optional)
            
        Returns:
            DataFrame with dataset information
        """
        query = "SELECT * FROM dataset_metadata"
        params = []
        
        if dataset_name:
            query += " WHERE dataset_name = ?"
            params.append(dataset_name)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def validate_data_integrity(self, dataset_name: str) -> Dict[str, any]:
        """Validate data integrity for a dataset.
        
        Args:
            dataset_name: Name of dataset to validate
            
        Returns:
            Dict with validation results
        """
        with sqlite3.connect(self.db_path) as conn:
            # Check for missing dates
            cursor = conn.execute("""
                SELECT MIN(date) as min_date, MAX(date) as max_date,
                       COUNT(DISTINCT date) as unique_dates
                FROM market_data WHERE dataset_name = ?
            """, (dataset_name,))
            date_stats = cursor.fetchone()
            
            # Check for null values
            cursor = conn.execute("""
                SELECT COUNT(*) as null_count
                FROM market_data 
                WHERE dataset_name = ? AND value IS NULL
            """, (dataset_name,))
            null_count = cursor.fetchone()[0]
            
            # Check for duplicate entries
            cursor = conn.execute("""
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT date || '|' || column_name) as unique_records
                FROM market_data WHERE dataset_name = ?
            """, (dataset_name,))
            record_stats = cursor.fetchone()
            
            return {
                'dataset_name': dataset_name,
                'date_range': (date_stats[0], date_stats[1]) if date_stats[0] else None,
                'unique_dates': date_stats[2],
                'null_values': null_count,
                'total_records': record_stats[0],
                'unique_records': record_stats[1],
                'has_duplicates': record_stats[0] != record_stats[1],
                'validation_timestamp': datetime.now().isoformat()
            }
    
    def get_update_summary(self, days: int = 30) -> pd.DataFrame:
        """Get summary of recent updates.
        
        Args:
            days: Number of days to look back
            
        Returns:
            DataFrame with update summary
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT dataset_name, operation, records_affected, 
                       source_file, timestamp
                FROM audit_log
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            """.format(days), conn)