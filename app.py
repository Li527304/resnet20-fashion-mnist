# 本地演示服务，只监听 127.0.0.1
import base64,io,json,webbrowser,argparse
from http.server import BaseHTTPRequestHandler,HTTPServer
from PIL import Image, ImageOps
import torch
from torchvision.datasets import FashionMNIST
from infer import ROOT,load,predict,CLASSES
from models import VARIANTS
from scene_crop import prepare

MODELS={}
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/':
            content=(ROOT/'demo.html').read_bytes();mime='text/html; charset=utf-8'
        elif self.path=='/api/summary':
            content=(ROOT/'results/summary.json').read_bytes();mime='application/json'
        elif self.path.startswith('/api/scene-sample/'):
            manifest=json.loads((ROOT/'results/scene/samples.json').read_text(encoding='utf-8'))
            i=int(self.path.rsplit('/',1)[-1])
            if not 0 <= i < len(manifest['samples']):
                self.send_error(404)
                return
            sample=manifest['samples'][i]
            raw=(ROOT/sample['path']).read_bytes()
            content=json.dumps({'image':'data:image/jpeg;base64,'+base64.b64encode(raw).decode(),
                                'label':sample['label'],'id':sample['id'],
                                'box':json.loads((ROOT/'results/scene/crop_protocol.json').read_text(encoding='utf-8'))['manual_boxes_xyxy'][str(sample['id'])]}).encode()
            mime='application/json'
        elif self.path.startswith('/api/sample/'):
            i=int(self.path.rsplit('/',1)[-1]);ds=FashionMNIST(ROOT/'data',train=False)
            image,label=ds[i];b=io.BytesIO();image.save(b,format='PNG')
            content=json.dumps({'image':'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode(),'label':CLASSES[label],'index':i}).encode();mime='application/json'
        elif self.path.startswith('/figures/') and '/' not in self.path[len('/figures/'):]:
            path=ROOT/self.path.lstrip('/');content=path.read_bytes();mime='image/png'
        else:
            self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
    def do_POST(self):
        if self.path!='/api/predict':self.send_error(404);return
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n>8*1024*1024:raise ValueError('图片过大')
            data=json.loads(self.rfile.read(n));variant=data.get('variant','base')
            if variant not in VARIANTS:raise ValueError('模型无效')
            if variant not in MODELS:MODELS[variant]=load(variant,17)
            image=Image.open(io.BytesIO(base64.b64decode(data['image'].split(',')[-1])))
            mode=data.get('mode','raw')
            if mode not in ('raw','invert','crop_fit','crop_grabcut'):
                raise ValueError('处理方式无效')
            if mode=='invert':
                image=ImageOps.invert(image.convert('RGB'))
            if mode in ('crop_fit','crop_grabcut'):
                box=data.get('box')
                if not isinstance(box,list) or len(box)!=4 or any(type(v) is not int for v in box):
                    raise ValueError('请先框选衣物，或填写四个整数坐标')
                left,top,right,bottom=box
                if not 0<=left<right<=image.width or not 0<=top<bottom<=image.height:
                    raise ValueError('裁剪框超出图片范围')
                image,_=prepare(image,box,mode=='crop_grabcut')
            result=predict(MODELS[variant],image)
            content=json.dumps(result,ensure_ascii=False).encode('utf-8');self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.end_headers();self.wfile.write(content)
        except Exception as e:
            self.send_response(400);self.end_headers();self.wfile.write(json.dumps({'error':str(e)}).encode())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);p.add_argument('--no-browser',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    server=HTTPServer(('127.0.0.1',a.port),Handler);print(f'http://127.0.0.1:{a.port}',flush=True)
    if not a.no_browser:webbrowser.open(f'http://127.0.0.1:{a.port}')
    server.serve_forever()
