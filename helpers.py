def _mk_result(best, thr_type):
                    return ModelEvaluationResult(
                        model_name=model_name,
                        split=split_name,
                        threshold_type=thr_type,
                        threshold=best['threshold'],
                        precision=best['precision'],
                        recall=best['recall'],
                        accuracy=best['accuracy'],
                        cost=best['cost'],
                        tp=best['tp'], fp=best['fp'],
                        tn=best['tn'], fn=best['fn'],
                        is_base_model=False
                    )
