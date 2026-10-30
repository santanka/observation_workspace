"""No external requests: selective granules, byte refresh, rollback history and dry-run."""
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from acquisition import Acquisition, unit
from test_acquisition import fixture, Response
import refresh_inputs as refresh

UTC=dt.timezone.utc


class RefreshTests(unittest.TestCase):
    def test_product_selection_and_boundaries(self):
        start=dt.datetime(2022,1,1,tzinfo=UTC);end=start+dt.timedelta(days=1)
        units=refresh.planned_units(start,end,['mgf64hz','mgf64hz'])
        self.assertEqual(len(units),26)
        self.assertEqual({p for p,_ in units},{'64hz'})
        particles=refresh.planned_units(start,end,['lepi'])
        self.assertEqual(len(particles),3)
        daily=refresh.planned_units(start,end,['mgf8sec'])
        self.assertEqual(len(daily),3)

    def test_refresh_only_selected_product_with_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=refresh.parser().parse_args(['--start','2022-09-29T05:05','--end','2022-09-29T05:10',
                 '--products','mgf64hz','--input-root',str(root/'raw'),'--output-root',str(root/'out'),
                 '--scratch-root',str(root/'scratch'),'--download-workers','1','--progress','off',
                 '--min-free-memory-gib','0','--scratch-reserve-gib','0','--remote-root','http://127.0.0.1/erg/'])
            start=dt.datetime(2022,9,29,5,5,tzinfo=UTC);end=start+dt.timedelta(minutes=5)
            planned=refresh.planned_units(start,end,args.products)
            u=unit('64hz',start.replace(minute=0));name=u['prefix']+'_v04.06.cdf'
            index_url=args.remote_root+u['folder']+'/'
            index=('<html><h1>Index of /erg/'+u['folder']+'</h1><a href="'+name+'">cdf</a></html>').encode()
            source=root/'fixture.cdf';fixture(source)
            first=source.read_bytes()
            unrelated=root/'raw/lepi/untouched.cdf';unrelated.parent.mkdir(parents=True);unrelated.write_bytes(b'untouched')
            def execute(body):
                calls=[]
                def request(url,method='GET'):
                    calls.append(url)
                    return Response(index if url==index_url else body,url)
                with patch.object(Acquisition,'request',side_effect=request):
                    self.assertEqual(refresh.run(args,planned),0)
                self.assertTrue(all('/mgf/' in u for u in calls))
            execute(first)
            target=root/'raw'/u['folder']/name
            before=target.stat().st_mtime_ns
            execute(first)
            self.assertEqual(target.stat().st_mtime_ns,before)
            fixture(source,conflict=True)
            updated=source.read_bytes();self.assertNotEqual(first,updated)
            execute(updated)
            self.assertEqual(target.read_bytes(),updated)
            self.assertEqual(unrelated.read_bytes(),b'untouched')
            history=list((root/'out/verified_inputs/history').rglob('*.cdf'))
            self.assertTrue(any(p.read_bytes()==first for p in history))
            report=json.loads((root/'out/verified_inputs/last_product_refresh.json').read_text())
            self.assertEqual(report['products'],['mgf64hz'])
            self.assertEqual(report['counts'],{'verified':1})

    def test_dry_run_never_constructs_acquisition(self):
        with patch('sys.argv',['refresh_inputs.py','--year','2022','--products','lepi','--dry-run']),\
             patch.object(refresh,'Acquisition',side_effect=AssertionError('No acquisition in dry-run')):
            self.assertEqual(refresh.main(),0)


if __name__=='__main__':unittest.main()
