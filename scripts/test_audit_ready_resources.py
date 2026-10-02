"""Integrity checks for measured totals and resource metadata."""
import copy
import unittest
from audit_ready_resources import verify_totals, verify_memory


class ResourceTotalsTests(unittest.TestCase):
    def setUp(self):
        self.report = dict(details=[dict(duration_s=2.,elapsed_s=1.,cpu_s=.5,status='ok')],
                           audio_s=2.,wall_s=1.,cpu_s=.5,rtf=.5,cpu_rtf=.25,failed=0)

    def test_correct_totals_pass(self):
        verify_totals(self.report)

    def test_reject_silent_failed_request_and_bad_rtf(self):
        for field, value in [('failed',1),('rtf',.7),('wall_s',2.)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_totals(dict(self.report,**{field:value}))

    def test_reject_nonfinite_and_negative_measurement(self):
        for value in [float('nan'), -1.]:
            bad=copy.deepcopy(self.report)
            bad['details'][0]['cpu_s']=value
            with self.assertRaises(ValueError):
                verify_totals(bad)


class MemorySamplingTests(unittest.TestCase):
    def test_nonatomic_psutil_rss_and_pss_readings_are_reported_not_rejected(self):
        sample = dict(samples=175, peak_rss_bytes=4847882240,
                      peak_pss_bytes=4906243072, peak_uss_bytes=4905959424,
                      last_pss_bytes=4520580096)
        verify_memory({'measured':sample})

    def test_negative_memory_and_peak_below_last_are_rejected(self):
        sample = dict(samples=1, peak_rss_bytes=100,peak_pss_bytes=90,
                      peak_uss_bytes=80,last_pss_bytes=90)
        for key,value in [('peak_rss_bytes',-1),('last_pss_bytes',91)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                verify_memory({'measured':dict(sample,**{key:value})})


if __name__ == '__main__':
    unittest.main()
