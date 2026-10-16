"""
qml_runtime.py - pickle-free inference for QSVM / VQC / QNN.
Needs only: qiskit, numpy, scikit-learn, joblib  (no qiskit-machine-learning, no dill).
Works on any Python version that can install qiskit, so Colab and Render no longer have to match.
"""
from pathlib import Path
import numpy as np
import joblib
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import Statevector

NQ, NCLS = 4, 3          # 4 qubits (4 PCA features), 3 potato classes


def qsvm_feature_map():
    return ZZFeatureMap(NQ, reps=2, entanglement="linear")


def build_vqc_circuit():
    """Same circuit as VQC / SamplerQNN in the notebook: ZZFeatureMap(reps=1) + RealAmplitudes(reps=3)."""
    fm = ZZFeatureMap(NQ, reps=1)
    an = RealAmplitudes(NQ, reps=3)
    qc = QuantumCircuit(NQ)
    qc.compose(fm, inplace=True)
    qc.compose(an, inplace=True)
    return qc, list(fm.parameters), list(an.parameters)


class QNNRuntime:
    """Used for BOTH VQC and QNN (identical circuit; interpret = bitstring % 3)."""
    def __init__(self, weights):
        self.w = np.asarray(weights, dtype=float).ravel()
        self.qc, self.xp, self.wp = build_vqc_circuit()
        if len(self.w) != len(self.wp):
            raise ValueError(f"weights has {len(self.w)} values, circuit needs {len(self.wp)}")
        self.idx = np.arange(2 ** NQ) % NCLS

    def predict_proba(self, X):
        out = []
        for x in np.asarray(X, dtype=float).reshape(-1, NQ):
            bind = {p: float(v) for p, v in zip(self.xp, x)}
            bind.update({p: float(v) for p, v in zip(self.wp, self.w)})
            probs = Statevector(self.qc.assign_parameters(bind)).probabilities()
            out.append(np.bincount(self.idx, weights=probs, minlength=NCLS))
        return np.array(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class QSVMRuntime:
    """Quantum kernel K(x, y) = |<phi(x)|phi(y)>|^2 computed exactly with statevectors + saved sklearn SVC."""
    def __init__(self, svc, Xtr):
        self.svc = svc
        self.fm = qsvm_feature_map()
        self.params = list(self.fm.parameters)
        self.Xtr = np.asarray(Xtr, dtype=float)
        self.T = np.array([self._state(x) for x in self.Xtr])      # train states, computed once

    def _state(self, x):
        bound = self.fm.assign_parameters({p: float(v) for p, v in zip(self.params, x)})
        return Statevector(bound).data

    def kernel(self, X):
        S = np.array([self._state(x) for x in np.asarray(X, dtype=float).reshape(-1, NQ)])
        return np.abs(S.conj() @ self.T.T) ** 2

    def predict(self, X):
        return self.svc.predict(self.kernel(X))

    def predict_proba(self, X):
        return self.svc.predict_proba(self.kernel(X))


def load(name, folder="."):
    folder = Path(folder)
    if name == "QSVM":
        return QSVMRuntime(joblib.load(folder / "qsvm_svc.pkl"), np.load(folder / "qsvm_train_X.npy"))
    if name == "VQC":
        return QNNRuntime(np.load(folder / "vqc_weights.npy"))
    if name == "QNN":
        return QNNRuntime(np.load(folder / "qnn_weights.npy"))
    raise ValueError(name)
