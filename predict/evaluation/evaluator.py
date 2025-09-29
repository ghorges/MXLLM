"""
结果评估模块
整合所有算法的结果并生成评估报告
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
import time
from pathlib import Path


class Evaluator:
    def __init__(self, output_dir: str = "./results"):
        """
        初始化评估器
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def evaluate_algorithms(self, algorithms: list, X_train, X_test, y_train, y_test, task_name: str) -> Dict[str, Any]:
        """
        Evaluate multiple algorithms
        
        Args:
            algorithms: List of algorithm instances
            X_train, X_test, y_train, y_test: Training and test data
            task_name: Task name for identification
            
        Returns:
            Dictionary of results for all algorithms
        """
        results = {}
        
        for algorithm in algorithms:
            try:
                print(f"   🔄 Training {algorithm.name}...")
                result = algorithm.fit_and_evaluate(X_train, y_train, X_test, y_test)
                results[algorithm.name] = result
                self.add_result(task_name, algorithm.name, result)
                
            except Exception as e:
                print(f"   ❌ {algorithm.name} failed: {e}")
                continue
        
        return results
    
    def add_result(self, task_name: str, algorithm_name: str, result: Dict[str, Any]):
        """
        添加算法结果
        
        Args:
            task_name: 任务名称（如rl_class, eab_class）
            algorithm_name: 算法名称
            result: 结果字典
        """
        if task_name not in self.results:
            self.results[task_name] = {}
        
        self.results[task_name][algorithm_name] = result
        
    def generate_summary_table(self, task_name: str) -> pd.DataFrame:
        """
        生成任务的汇总表
        
        Args:
            task_name: 任务名称
            
        Returns:
            汇总表DataFrame
        """
        if task_name not in self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        for algo_name, result in self.results[task_name].items():
            summary_data.append({
                'Algorithm': algo_name,
                'Accuracy': result.get('accuracy', 0),
                'Precision': result.get('precision', 0),
                'Recall': result.get('recall', 0),
                'F1_Score': result.get('f1_score', 0),
                'Training_Time': result.get('training_time', 0),
                'Prediction_Time': result.get('prediction_time', 0)
            })
        
        df = pd.DataFrame(summary_data)
        
        # 按F1分数排序
        if not df.empty:
            df = df.sort_values('F1_Score', ascending=False).reset_index(drop=True)
        
        return df
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        生成比较报告
        
        Returns:
            比较报告字典
        """
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tasks': {},
            'overall_best': {}
        }
        
        for task_name in self.results.keys():
            summary_df = self.generate_summary_table(task_name)
            
            if summary_df.empty:
                continue
            
            # 任务详细信息
            task_info = {
                'summary_table': summary_df.to_dict('records'),
                'best_algorithm': summary_df.iloc[0]['Algorithm'] if len(summary_df) > 0 else None,
                'best_f1_score': summary_df.iloc[0]['F1_Score'] if len(summary_df) > 0 else 0,
                'algorithm_count': len(summary_df),
                'metrics_comparison': self._generate_metrics_comparison(task_name)
            }
            
            report['tasks'][task_name] = task_info
        
        # 总体最佳算法
        self._find_overall_best(report)
        
        return report
    
    def _generate_metrics_comparison(self, task_name: str) -> Dict[str, Dict[str, float]]:
        """
        生成指标比较
        
        Args:
            task_name: 任务名称
            
        Returns:
            指标比较字典
        """
        comparison = {
            'accuracy': {},
            'precision': {},
            'recall': {},
            'f1_score': {},
            'training_time': {},
            'prediction_time': {}
        }
        
        for algo_name, result in self.results[task_name].items():
            for metric in comparison.keys():
                comparison[metric][algo_name] = result.get(metric, 0)
        
        return comparison
    
    def _find_overall_best(self, report: Dict[str, Any]):
        """
        找出总体最佳算法
        
        Args:
            report: 报告字典
        """
        best_by_task = {}
        
        for task_name, task_info in report['tasks'].items():
            if task_info['best_algorithm']:
                best_by_task[task_name] = {
                    'algorithm': task_info['best_algorithm'],
                    'f1_score': task_info['best_f1_score']
                }
        
        # 按任务计算平均F1分数
        algo_scores = {}
        for task_name, best_info in best_by_task.items():
            algo_name = best_info['algorithm']
            f1_score = best_info['f1_score']
            
            if algo_name not in algo_scores:
                algo_scores[algo_name] = []
            algo_scores[algo_name].append(f1_score)
        
        # 计算平均分数
        algo_avg_scores = {}
        for algo_name, scores in algo_scores.items():
            algo_avg_scores[algo_name] = np.mean(scores)
        
        if algo_avg_scores:
            best_algo = max(algo_avg_scores, key=algo_avg_scores.get)
            report['overall_best'] = {
                'algorithm': best_algo,
                'average_f1_score': algo_avg_scores[best_algo],
                'task_wins': len([t for t, info in best_by_task.items() 
                                if info['algorithm'] == best_algo])
            }
    
    def save_results(self, results: Dict[str, Any] = None, task_name: str = None, output_dir: str = None, filename: str = "evaluation_results.json"):
        """
        Save results to file (overloaded method)
        
        Args:
            results: Results dictionary (optional)
            task_name: Task name (optional)
            output_dir: Output directory (optional)
            filename: Filename
        """
        if results is not None:
            # New signature: save specific results
            import os
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{task_name}_{filename}")
            else:
                output_path = self.output_dir / f"{task_name}_{filename}"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ Results saved to: {output_path}")
        else:
            # Original signature: save all results
            output_path = self.output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ Detailed results saved to: {output_path}")
    
    def generate_summary_report(self, all_results: Dict[str, Dict[str, Any]], output_dir: str):
        """
        Generate summary report for all tasks
        
        Args:
            all_results: Dictionary of all task results
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        summary = {}
        for task_name, task_results in all_results.items():
            summary[task_name] = {}
            for alg_name, result in task_results.items():
                summary[task_name][alg_name] = {
                    'accuracy': result.get('accuracy', 0),
                    'f1_score': result.get('f1_score', 0),
                    'precision': result.get('precision', 0),
                    'recall': result.get('recall', 0),
                    'training_time': result.get('training_time', 0),
                    'prediction_time': result.get('prediction_time', 0)
                }
        
        summary_path = os.path.join(output_dir, "summary_report.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Summary report saved to: {summary_path}")
        
        # Print summary to console
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 80)
        for task_name, task_results in summary.items():
            print(f"\n🎯 {task_name.upper()}:")
            for alg_name, metrics in task_results.items():
                print(f"   {alg_name:15s} | Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1_score']:.3f} | Time: {metrics['training_time']:.2f}s")
    
    def save_summary_tables(self):
        """保存汇总表到CSV文件"""
        for task_name in self.results.keys():
            summary_df = self.generate_summary_table(task_name)
            
            if not summary_df.empty:
                output_path = self.output_dir / f"{task_name}_summary.csv"
                summary_df.to_csv(output_path, index=False)
                print(f"✅ {task_name} 汇总表已保存到: {output_path}")
    
    def print_summary(self):
        """打印汇总信息"""
        print("\n" + "="*80)
        print("📊 预测结果汇总")
        print("="*80)
        
        for task_name in self.results.keys():
            print(f"\n🎯 任务: {task_name}")
            print("-" * 50)
            
            summary_df = self.generate_summary_table(task_name)
            
            if summary_df.empty:
                print("   ❌ 没有有效结果")
                continue
            
            # 显示前三名
            top_3 = summary_df.head(3)
            for i, row in top_3.iterrows():
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"   {rank} {row['Algorithm']}: "
                      f"F1={row['F1_Score']:.4f}, "
                      f"Acc={row['Accuracy']:.4f}, "
                      f"Time={row['Training_Time']:.2f}s")
        
        # 总体最佳
        report = self.generate_comparison_report()
        if 'overall_best' in report and report['overall_best']:
            best = report['overall_best']
            print(f"\n🏆 总体最佳算法: {best['algorithm']}")
            print(f"   平均F1分数: {best['average_f1_score']:.4f}")
            print(f"   获胜任务数: {best['task_wins']}")
        
        print("\n" + "="*80)
    
    def create_visualization(self):
        """创建可视化图表"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('default')
            
            for task_name in self.results.keys():
                summary_df = self.generate_summary_table(task_name)
                
                if summary_df.empty:
                    continue
                
                # 创建对比图
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'{task_name} 算法性能对比', fontsize=16, fontweight='bold')
                
                # 准确率对比
                axes[0, 0].bar(summary_df['Algorithm'], summary_df['Accuracy'])
                axes[0, 0].set_title('准确率对比')
                axes[0, 0].set_ylabel('准确率')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # F1分数对比
                axes[0, 1].bar(summary_df['Algorithm'], summary_df['F1_Score'])
                axes[0, 1].set_title('F1分数对比')
                axes[0, 1].set_ylabel('F1分数')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
                # 训练时间对比
                axes[1, 0].bar(summary_df['Algorithm'], summary_df['Training_Time'])
                axes[1, 0].set_title('训练时间对比')
                axes[1, 0].set_ylabel('时间(秒)')
                axes[1, 0].tick_params(axis='x', rotation=45)
                
                # 综合雷达图（如果算法数量较少）
                if len(summary_df) <= 5:
                    angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
                    angles = np.concatenate((angles, [angles[0]]))
                    
                    axes[1, 1].remove()
                    ax_radar = fig.add_subplot(224, projection='polar')
                    
                    for i, row in summary_df.iterrows():
                        values = [row['Accuracy'], row['Precision'], 
                                row['Recall'], row['F1_Score']]
                        values = np.concatenate((values, [values[0]]))
                        ax_radar.plot(angles, values, 'o-', linewidth=2, 
                                    label=row['Algorithm'])
                    
                    ax_radar.set_xticks(angles[:-1])
                    ax_radar.set_xticklabels(['准确率', '精确率', '召回率', 'F1分数'])
                    ax_radar.set_title('性能雷达图')
                    ax_radar.legend()
                else:
                    # 精确率vs召回率散点图
                    axes[1, 1].scatter(summary_df['Precision'], summary_df['Recall'])
                    for i, row in summary_df.iterrows():
                        axes[1, 1].annotate(row['Algorithm'], 
                                          (row['Precision'], row['Recall']))
                    axes[1, 1].set_xlabel('精确率')
                    axes[1, 1].set_ylabel('召回率')
                    axes[1, 1].set_title('精确率 vs 召回率')
                
                plt.tight_layout()
                
                # 保存图像
                output_path = self.output_dir / f"{task_name}_comparison.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✅ {task_name} 对比图已保存到: {output_path}")
                plt.close()
                
        except ImportError:
            print("⚠️ matplotlib或seaborn未安装，跳过可视化")
        except Exception as e:
            print(f"❌ 创建可视化时出错: {e}")


if __name__ == "__main__":
    # 测试评估器
    evaluator = Evaluator()
    
    # 模拟一些结果
    test_results = {
        'PLS': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1_score': 0.85, 'training_time': 2.5},
        'RF': {'accuracy': 0.90, 'precision': 0.89, 'recall': 0.91, 'f1_score': 0.90, 'training_time': 5.2},
        'MLP': {'accuracy': 0.87, 'precision': 0.85, 'recall': 0.89, 'f1_score': 0.87, 'training_time': 15.8}
    }
    
    for algo_name, result in test_results.items():
        evaluator.add_result('rl_class', algo_name, result)
    
    # 生成报告
    evaluator.print_summary()
    evaluator.save_summary_report()
    evaluator.save_summary_tables()
    
    print("评估器测试完成！") 