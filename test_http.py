import json
import re
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from server import Handler, ThreadingHTTPServer
from catalog import PROBLEMS


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=cls.server.serve_forever,daemon=True).start()
        cls.base=f'http://127.0.0.1:{cls.server.server_port}'
        html=urlopen(cls.base).read().decode()
        cls.token=re.search(r'name="lab-token" content="([^"]+)"',html).group(1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close()

    def request(self,path,data=None,token=True):
        headers={'Content-Type':'application/json'}
        if token: headers['X-Lab-Token']=self.token
        req=Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)
        with urlopen(req,timeout=5) as response: return json.load(response)

    def wait(self,key):
        end=time.monotonic()+40
        while time.monotonic()<end:
            job=self.request('/api/job/'+key)
            if job['status']=='done': return job['result']
            time.sleep(0.1)
        self.fail('Job did not finish')

    def test_submit_and_disclosure(self):
        p=self.request('/api/problem/ppo_loss');self.assertNotIn('reference',p)
        job=self.request('/api/run',dict(problem='ppo_loss',code=PROBLEMS['ppo_loss']['reference'],mode='submit'))
        r=self.wait(job['id']);self.assertTrue(r['passed'],r);self.assertEqual(len(r['cases']),9)

    def test_cross_site_post_rejected(self):
        with self.assertRaises(HTTPError) as e: self.request('/api/run',{},token=False)
        self.assertEqual(e.exception.code,403)
        e.exception.close()

    def test_cancel_and_timeout(self):
        request=dict(problem='softmax',code='def solve(x):\n while True: pass',mode='run')
        job=self.request('/api/run',request)
        self.request('/api/cancel/'+job['id'],{})
        self.assertIn('停止',self.wait(job['id'])['error'])
        job=self.request('/api/run',request)
        self.assertIn('30 秒',self.wait(job['id'])['error'])

    def test_unicode_debug(self):
        job=self.request('/api/run',dict(problem='softmax',code='def solve(x): return x',mode='debug',custom='print("中文调试正常")'))
        r=self.wait(job['id']);self.assertTrue(r['passed']);self.assertIn('中文调试正常',r['stdout'])


if __name__=='__main__':unittest.main(verbosity=2)
