# tests/integration/orchestrator.py
import pytest
from pathlib import Path
import time
import json
from typing import List, Dict, Any
from datetime import datetime

class IntegrationTestOrchestrator:
    """Orchestrate integration test execution."""
    
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.report_dir.mkdir(exist_ok=True)
        self.results = []
        
    def run_test_suite(self, mode: str = "fast") -> Dict[str, Any]:
        """Run integration test suite.
        
        Args:
            mode: "fast" for smoke tests only, "thorough" for all tests
        """
        start_time = time.time()
        
        if mode == "fast":
            test_modules = [
                "tests/integration/test_workflow_simulation.py::TestWorkflowSimulation::test_mini_pipeline",
                "tests/integration/test_real_data_smoke.py::TestRealDataSmoke::test_small_region_processing"
            ]
        else:
            test_modules = [
                "tests/integration/test_workflow_simulation.py",
                "tests/integration/test_system_limits.py",
                "tests/integration/test_edge_cases.py",
                "tests/integration/test_real_data_smoke.py"
            ]
        
        # Run tests and collect results
        for module in test_modules:
            result = self._run_test_module(module)
            self.results.append(result)
        
        # Generate report
        duration = time.time() - start_time
        report = self._generate_report(duration)
        
        # Save report
        report_path = self.report_dir / f"integration_test_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _run_test_module(self, module: str) -> Dict[str, Any]:
        """Run a single test module."""
        start_time = time.time()
        
        # Run pytest programmatically
        exit_code = pytest.main([
            module,
            "-v",
            "--tb=short",
            f"--junit-xml={self.report_dir}/junit_{Path(module).stem}.xml"
        ])
        
        duration = time.time() - start_time
        
        return {
            'module': module,
            'success': exit_code == 0,
            'duration': duration,
            'exit_code': exit_code
        }
    
    def _generate_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate test report."""
        passed = sum(1 for r in self.results if r['success'])
        failed = len(self.results) - passed
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_duration': total_duration,
            'summary': {
                'total': len(self.results),
                'passed': passed,
                'failed': failed
            },
            'results': self.results,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for failures
        failures = [r for r in self.results if not r['success']]
        if failures:
            recommendations.append(f"Fix {len(failures)} failing test modules before processing real data")
            for failure in failures:
                recommendations.append(f"  - Fix: {failure['module']}")
        
        # Check for slow tests
        slow_tests = [r for r in self.results if r['duration'] > 300]  # 5 minutes
        if slow_tests:
            recommendations.append(f"Optimize {len(slow_tests)} slow test modules")
        
        if not recommendations:
            recommendations.append("All tests passed! System ready for real data processing.")
        
        return recommendations


# Main execution script
if __name__ == "__main__":
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "fast"
    
    orchestrator = IntegrationTestOrchestrator(Path("test_reports"))
    report = orchestrator.run_test_suite(mode)
    
    print("\n" + "="*60)
    print("Integration Test Report")
    print("="*60)
    print(f"Total Duration: {report['total_duration']:.1f} seconds")
    print(f"Tests Passed: {report['summary']['passed']}/{report['summary']['total']}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")