"""Convenience reporting helpers for the Hydrangea analysis pipeline."""


def report_normalization(obj, show=True):
    return obj.normalization_report(show=show)


def report_embeddings(obj, show=True, scope="global", label=None):
    return obj.report_embeddings(scope=scope, label=label, show=show)


def report_hmm(obj, report="selected"):
    return obj.hmm_report(report=report)


def report_all(obj, normalization=True, embeddings=True, hmm="selected"):
    results = {}
    if normalization:
        results["normalization"] = obj.normalization_report(show=True)
    if embeddings:
        results["embeddings"] = obj.report_embeddings(show=True)
    if hmm is not None:
        results["hmm"] = obj.hmm_report(report=hmm)
    return results