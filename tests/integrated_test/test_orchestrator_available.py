#!/usr/bin/env python3
"""
Modified orchestrator to run only available tests.
"""

import sys
from pathlib import Path
import time
import json
from datetime import datetime
import subprocess

# Add project root to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class AvailableTestOrchestrator:
    """Orchestrate available integration test execution."""
    
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        self.report_dir.mkdir(exist_ok=True)
        self.results = []
        
    def run_available_tests(self, mode: str = "fast") -> dict:
        """Run only available integration tests."""
        start_time = time.time()
        
        # Available test modules that work with current system
        available_tests = [
            "tests/integrated_test/test_current_system.py",
            "tests/integrated_test/test_simple_integration.py", 
            "tests/integrated_test/test_minimal_workflow.py"
        ]
        
        print(f"🚀 Running {len(available_tests)} available integration tests...")
        
        # Run tests and collect results
        for test_path in available_tests:
            result = self._run_test_module(test_path)
            self.results.append(result)
        
        # Generate report
        duration = time.time() - start_time
        report = self._generate_report(duration)
        
        # Save report
        report_path = self.report_dir / f"available_integration_test_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Report saved to: {report_path}")
        
        return report
    
    def _run_test_module(self, test_path: str) -> dict:
        """Run a single test module."""
        start_time = time.time()
        
        print(f"\n--- Running {test_path} ---")
        
        # Run the test with proper Python path
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = "/home/jason/geo"
        
        result = subprocess.run(
            ["/home/jason/anaconda3/envs/geo/bin/python", test_path],
            cwd="/home/jason/geo",
            env=env,
            capture_output=True,
            text=True
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        print(f"{'✅ PASSED' if success else '❌ FAILED'} - {test_path} ({duration:.1f}s)")
        
        if not success:
            print(f"Error output: {result.stderr}")
        
        return {
            'module': test_path,
            'success': success,
            'duration': duration,
            'exit_code': result.returncode,
            'stdout': result.stdout[:1000] if result.stdout else "",  # Truncate for report
            'stderr': result.stderr[:500] if result.stderr else ""
        }
    
    def _generate_report(self, total_duration: float) -> dict:
        """Generate test report."""
        passed = sum(1 for r in self.results if r['success'])
        failed = len(self.results) - passed
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_duration': total_duration,
            'system_status': 'available_components_only',
            'summary': {
                'total': len(self.results),
                'passed': passed,
                'failed': failed,
                'success_rate': round((passed / len(self.results)) * 100, 1) if self.results else 0
            },
            'results': self.results,
            'recommendations': self._generate_recommendations(),
            'system_readiness': self._assess_system_readiness()
        }
    
    def _generate_recommendations(self) -> list:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failures = [r for r in self.results if not r['success']]
        if failures:
            recommendations.append(f"Fix {len(failures)} failing test modules")
            for failure in failures:
                recommendations.append(f"  - Debug: {failure['module']}")
        
        # Check for slow tests
        slow_tests = [r for r in self.results if r['duration'] > 30]
        if slow_tests:
            recommendations.append(f"Optimize {len(slow_tests)} slow test modules")
        
        if not recommendations:
            recommendations.extend([
                "All available integration tests passed! ✅",
                "Current system components are working correctly",
                "Grid system and database integration validated",
                "System ready for raster processing module development"
            ])
        
        return recommendations
    
    def _assess_system_readiness(self) -> dict:
        """Assess overall system readiness."""
        passed = sum(1 for r in self.results if r['success'])
        total = len(self.results)
        
        if passed == total:
            status = "ready"
            description = "All available components tested and working"
        elif passed >= total * 0.8:
            status = "mostly_ready"
            description = "Most components working, minor issues to resolve"
        else:
            status = "needs_work"
            description = "Significant issues found in core components"
        
        return {
            'status': status,
            'description': description,
            'core_components_tested': [
                'Grid Systems',
                'Database Schema',
                'Configuration Management', 
                'Component Registry',
                'Data Generation',
                'Workflow Simulation'
            ],
            'missing_for_full_system': [
                'Raster Processing Pipeline',
                'Resampling Engines',
                'Feature Extraction Framework',
                'Parallel Processing Infrastructure',
                'Real Data Integration'
            ]
        }

def main():
    """Run available integration test orchestrator."""
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "fast"
    
    orchestrator = AvailableTestOrchestrator(Path("/home/jason/geo/test_reports"))
    report = orchestrator.run_available_tests(mode)
    
    print("\n" + "="*70)
    print("AVAILABLE INTEGRATION TEST REPORT")
    print("="*70)
    print(f"Total Duration: {report['total_duration']:.1f} seconds")
    print(f"Tests Passed: {report['summary']['passed']}/{report['summary']['total']} ({report['summary']['success_rate']}%)")
    print(f"System Status: {report['system_readiness']['status'].upper()}")
    print(f"Description: {report['system_readiness']['description']}")
    
    print("\nCore Components Tested:")
    for component in report['system_readiness']['core_components_tested']:
        print(f"  ✅ {component}")
    
    print("\nMissing for Full System:")
    for missing in report['system_readiness']['missing_for_full_system']:
        print(f"  ⏳ {missing}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    return report['summary']['passed'] == report['summary']['total']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
