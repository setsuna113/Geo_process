# src/database/analysis_store.py
"""Database operations for spatial analysis results."""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import numpy as np
from pathlib import Path

from src.database.connection import DatabaseManager
from src.abstractions.interfaces.analyzer import AnalysisResult, AnalysisMetadata

logger = logging.getLogger(__name__)

class AnalysisStore:
    """Handle database storage and retrieval of analysis results."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Ensure analysis result tables exist."""
        create_experiments_sql = """
        CREATE TABLE IF NOT EXISTS analysis_experiments (
            id SERIAL PRIMARY KEY,
            experiment_name VARCHAR(255) UNIQUE NOT NULL,
            analysis_type VARCHAR(100) NOT NULL,
            input_shape INTEGER[] NOT NULL,
            input_bands TEXT[] NOT NULL,
            parameters JSONB NOT NULL,
            processing_time FLOAT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            data_source VARCHAR(500),
            normalization_applied BOOLEAN DEFAULT FALSE,
            coordinate_system VARCHAR(50) DEFAULT 'EPSG:4326',
            output_path VARCHAR(500),
            status VARCHAR(50) DEFAULT 'completed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_analysis_experiments_type 
        ON analysis_experiments(analysis_type);
        
        CREATE INDEX IF NOT EXISTS idx_analysis_experiments_timestamp 
        ON analysis_experiments(timestamp);
        """
        
        create_statistics_sql = """
        CREATE TABLE IF NOT EXISTS analysis_statistics (
            id SERIAL PRIMARY KEY,
            experiment_id INTEGER REFERENCES analysis_experiments(id) ON DELETE CASCADE,
            statistic_name VARCHAR(100) NOT NULL,
            statistic_value JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_analysis_statistics_experiment 
        ON analysis_statistics(experiment_id);
        """
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_experiments_sql)
                cur.execute(create_statistics_sql)
            conn.commit()
    
    def store_result(self, result: AnalysisResult, experiment_name: str, 
                    output_path: Optional[str] = None) -> int:
        """
        Store analysis result in database.
        
        Args:
            result: Analysis result to store
            experiment_name: Unique name for this experiment
            output_path: Path where files are stored
            
        Returns:
            Experiment ID
        """
        # Insert experiment record
        experiment_sql = """
        INSERT INTO analysis_experiments (
            experiment_name, analysis_type, input_shape, input_bands,
            parameters, processing_time, timestamp, data_source,
            normalization_applied, coordinate_system, output_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(experiment_sql, (
                    experiment_name,
                    result.metadata.analysis_type,
                    list(result.metadata.input_shape),
                    result.metadata.input_bands,
                    json.dumps(result.metadata.parameters),
                    result.metadata.processing_time,
                    datetime.fromisoformat(result.metadata.timestamp.replace(' ', 'T')),
                    result.metadata.data_source,
                    result.metadata.normalization_applied,
                    result.metadata.coordinate_system,
                    str(output_path) if output_path else None
                ))
                
                experiment_id = cur.fetchone()[0]
                
                # Store statistics
                if result.statistics:
                    self._store_statistics(cur, experiment_id, result.statistics)
                
            conn.commit()
            
        logger.info(f"Stored analysis result with experiment_id: {experiment_id}")
        return experiment_id
    
    def _store_statistics(self, cursor, experiment_id: int, statistics: Dict[str, Any]):
        """Store statistics for an experiment."""
        statistics_sql = """
        INSERT INTO analysis_statistics (experiment_id, statistic_name, statistic_value)
        VALUES (%s, %s, %s);
        """
        
        for stat_name, stat_value in statistics.items():
            # Convert numpy types to Python types for JSON serialization
            if isinstance(stat_value, np.ndarray):
                stat_value = stat_value.tolist()
            elif isinstance(stat_value, (np.integer, np.floating)):
                stat_value = stat_value.item()
            
            cursor.execute(statistics_sql, (experiment_id, stat_name, json.dumps(stat_value)))
    
    def get_experiment(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get experiment by name."""
        sql = """
        SELECT id, experiment_name, analysis_type, input_shape, input_bands,
               parameters, processing_time, timestamp, data_source,
               normalization_applied, coordinate_system, output_path, status
        FROM analysis_experiments 
        WHERE experiment_name = %s;
        """
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (experiment_name,))
                row = cur.fetchone()
                
                if not row:
                    return None
                
                experiment = {
                    'id': row[0],
                    'experiment_name': row[1],
                    'analysis_type': row[2],
                    'input_shape': row[3],
                    'input_bands': row[4],
                    'parameters': json.loads(row[5]) if row[5] else {},
                    'processing_time': row[6],
                    'timestamp': row[7],
                    'data_source': row[8],
                    'normalization_applied': row[9],
                    'coordinate_system': row[10],
                    'output_path': row[11],
                    'status': row[12]
                }
                
                # Get statistics
                experiment['statistics'] = self._get_statistics(cur, experiment['id'])
                
                return experiment
    
    def _get_statistics(self, cursor, experiment_id: int) -> Dict[str, Any]:
        """Get statistics for an experiment."""
        sql = """
        SELECT statistic_name, statistic_value 
        FROM analysis_statistics 
        WHERE experiment_id = %s;
        """
        
        cursor.execute(sql, (experiment_id,))
        rows = cursor.fetchall()
        
        statistics = {}
        for name, value in rows:
            statistics[name] = json.loads(value) if value else None
            
        return statistics
    
    def list_experiments(self, analysis_type: Optional[str] = None, 
                        limit: int = 50) -> List[Dict[str, Any]]:
        """List recent experiments."""
        if analysis_type:
            sql = """
            SELECT id, experiment_name, analysis_type, timestamp, status, output_path
            FROM analysis_experiments 
            WHERE analysis_type = %s
            ORDER BY timestamp DESC LIMIT %s;
            """
            params = (analysis_type, limit)
        else:
            sql = """
            SELECT id, experiment_name, analysis_type, timestamp, status, output_path
            FROM analysis_experiments 
            ORDER BY timestamp DESC LIMIT %s;
            """
            params = (limit,)
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                
                return [{
                    'id': row[0],
                    'experiment_name': row[1], 
                    'analysis_type': row[2],
                    'timestamp': row[3],
                    'status': row[4],
                    'output_path': row[5]
                } for row in rows]
    
    def delete_experiment(self, experiment_name: str) -> bool:
        """Delete an experiment and its statistics."""
        sql = "DELETE FROM analysis_experiments WHERE experiment_name = %s;"
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (experiment_name,))
                deleted = cur.rowcount > 0
            conn.commit()
            
        if deleted:
            logger.info(f"Deleted experiment: {experiment_name}")
        
        return deleted
    
    def update_experiment_status(self, experiment_name: str, status: str):
        """Update experiment status."""
        sql = "UPDATE analysis_experiments SET status = %s WHERE experiment_name = %s;"
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (status, experiment_name))
            conn.commit()