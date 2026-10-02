"""Regression checks for committed results and failed-recognition accounting."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import benchmark_ready_asr as runner


class DurableRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        audio = self.root / 'clip.wav'
        sf.write(audio, np.zeros(4000,dtype=np.float32),16000,subtype='FLOAT')
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps({'samples':[dict(id='one',path=str(audio),
            sha256=runner.sha256(audio),reference='сәлем әлем',language='kk',dataset='synthetic',
            split='test',condition='original')]}))
        self.output = self.root/'result.json'
        self.calls = 0
        self.fail = False
        def transcribe(audio):
            self.calls += 1
            if self.fail and self.calls > 1:
                raise RuntimeError('Synthetic inference failure')
            return {'hypothesis':'сәлем әлем'}
        model=SimpleNamespace(metadata={'frozen':'fake test backend'},transcribe=transcribe)
        self.backend=SimpleNamespace(__file__=__file__,ReadyASR=lambda *a,**k:model)

    def run_benchmark(self):
        argv=['runner','--backend','whisper','--model-dir',str(self.root),
              '--manifest',str(self.manifest),'--output',str(self.output)]
        with patch.object(sys,'argv',argv),patch.object(runner.importlib,'import_module',return_value=self.backend):
            runner.main()

    def test_complete_resume_does_not_infer_again(self):
        self.run_benchmark()
        self.assertEqual(self.calls,2)
        first=self.output.read_bytes()
        self.run_benchmark()
        self.assertEqual(self.calls,2)
        self.assertEqual(first,self.output.read_bytes())

    def test_stale_final_is_rejected(self):
        self.run_benchmark()
        result=json.loads(self.output.read_text())
        result['details'][0]['hypothesis']='changed'
        self.output.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError,'Final result differs'):
            self.run_benchmark()

    def test_failed_request_remains_as_deletions(self):
        self.fail=True
        self.run_benchmark()
        result=json.loads(self.output.read_text())
        self.assertEqual(result['summary']['failed_requests'],1)
        self.assertEqual(result['summary']['deletions'],2)
        self.assertEqual(result['summary']['wer_pct'],100)
        self.assertEqual(result['details'][0]['hypothesis'],'')
        self.run_benchmark()
        self.assertEqual(self.calls,2)


if __name__=='__main__':
    unittest.main()
