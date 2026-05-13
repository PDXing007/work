#!/usr/bin/env python3
"""
SSQ Predictor — Kivy 移动端应用 (美化版)
三屏：主页 | 历史开奖 | AI预测
自适应 P20(1080x2244) / K50P(1440x3200)
"""

import os, sys, json, math, threading
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.metrics import dp, sp
from kivy.graphics import Color, Rectangle, RoundedRectangle, Ellipse, Line
from kivy.utils import get_color_from_hex

# ---- Font ----
_CJK_FONT = None
for _fp in [
    "DroidSansFallback.ttf",                      # bundled in APK
    "C:/Windows/Fonts/msyh.ttc",                  # Windows dev
    "/system/fonts/DroidSansChinese.ttf",          # Huawei EMUI
    "/system/fonts/HwChinese-Medium.ttf",          # Huawei EMUI v2
    "/system/fonts/DroidSansFallback.ttf",         # AOSP / Samsung
    "/system/fonts/NotoSansSC-Regular.otf",        # Android 7+
    "/system/fonts/NotoSansCJK-Regular.ttc",       # Some custom ROMs
    "/system/fonts/MiLanProVF.ttf",                # MIUI (Xiaomi/Redmi)
    "/system/fonts/Roboto-Regular.ttf",            # fallback - no CJK
]:
    try:
        LabelBase.register("_cjk",_fp)
        LabelBase._fonts["Roboto"] = LabelBase._fonts["_cjk"]
        _CJK_FONT = _fp
        break
    except Exception:
        pass

# ---- Colors ----
BG   = get_color_from_hex("#0A0E14")
SURF = get_color_from_hex("#151B23")
SURF2= get_color_from_hex("#1C2533")
BORD = get_color_from_hex("#2D3540")
RED  = get_color_from_hex("#E84040")
BLUE = get_color_from_hex("#4488FF")
GOLD = get_color_from_hex("#F0B90B")
GRN  = get_color_from_hex("#28A745")
TXT  = get_color_from_hex("#E6EDF3")
TXT2 = get_color_from_hex("#768390")
WHT  = (1,1,1,1)

from data_loader import load_history, load_parsed, get_data_summary
from predictor import SSQPredictor

# ============= Data =============
class DataManager:
    def __init__(self):
        self.lock=threading.Lock(); self.parsed=[]; self.summary=None; self.status_msg=""
    def load(self):
        with self.lock:
            try:
                self.parsed=load_parsed()
                self.summary=get_data_summary(load_history())
            except: self.summary={"总期数":0,"最新一期":"N/A","日期范围":""}
            return self.summary
    def get_last_n(self,n=10):
        with self.lock:
            if not self.parsed: self.load()
            return self.parsed[:n] if self.parsed else []
    def refresh(self):
        try:
            import requests, traceback, os
            data=[]
            try: data=load_history()
            except: data=[]
            s=requests.Session(); s.trust_env=True
            r=s.get("https://jc.zhcw.com/port/client_json.php",
                    params={"transactionType":"10001003","lotteryId":"1","count":"10"},
                    headers={"User-Agent":"Mozilla/5.0"},timeout=15)
            api=r.json()
            if api.get("resCode")!="000000": self.status_msg="API错误"; return False
            exist={d["期号"] for d in data}
            missing=[i for i in api["issue"] if i not in exist]
            if not missing: self.status_msg=f"已是最新({len(data)}期)"; return True
            missing.reverse()
            for i,iss in enumerate(missing):
                self.status_msg=f"抓取{i+1}/{len(missing)}"
                r2=s.get("https://jc.zhcw.com/port/client_json.php",
                        params={"transactionType":"10001002","lotteryId":"1","issue":iss},
                        headers={"User-Agent":"Mozilla/5.0"},timeout=15)
                if r2.status_code==200:
                    d=r2.json(); rd=d.get("frontWinningNum",""); bl=d.get("backWinningNum","")
                    if rd and bl: data.append({"期号":iss,"开奖日期":d.get("openTime",""),"红球":rd,"蓝球":bl.strip()})
            data.sort(key=lambda x:x["期号"],reverse=True)
            # Save to multiple possible locations
            saved=False
            for p in [os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","ssq_全历史.json"),
                      os.path.join(os.getcwd(),"ssq_全历史.json"),
                      os.path.join(os.getcwd(),"..","ssq_全历史.json")]:
                try:
                    p=os.path.normpath(p)
                    with open(p,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False)
                    saved=True; break
                except: pass
            if not saved: self.status_msg="保存失败"; return False
            self.load(); self.status_msg=f"已更新{len(data)}期"; return True
        except Exception as e:
            self.status_msg=f"网络错误"; return False

# ============= Ball Widget =============
class Ball(FloatLayout):
    def __init__(self,number=0,is_blue=False,sz=None,**kw):
        super().__init__(**kw)
        self._sz=sz or dp(38); self.number=number; self.is_blue=is_blue
        self.size_hint=(None,None); self.size=(self._sz,self._sz)
        fs=sp(14) if self._sz<dp(40) else sp(16)
        self.add_widget(Label(text=str(number).zfill(2),font_size=fs,bold=True,color=WHT,
                              pos_hint={'center_x':.5,'center_y':.5},
                              size_hint=(None,None),size=(self._sz,self._sz),halign="center",valign="middle"))
        self.bind(pos=self._draw,size=self._draw)
    def _draw(self,*a):
        try:
            self.canvas.before.clear(); sz=self._sz; cx,cy=self.x+sz/2,self.y+sz/2; r=sz/2-dp(1)
            if r<=0: return
            c=BLUE if self.is_blue else RED
            with self.canvas.before:
                Color(*c); Ellipse(pos=(cx-r,cy-r),size=(r*2,r*2))
                Color(1,1,1,.25); Ellipse(pos=(cx-r*.5,cy-r*.05),size=(r*1.0,r*.6))
        except Exception: pass

# ============= Card Button =============
class CardBtn(FloatLayout):
    def __init__(self,title,desc,color,on_press=None,**kw):
        super().__init__(**kw)
        self.size_hint_y=None; self.height=dp(70)
        with self.canvas.before:
            Color(*SURF)
            self._bg=RoundedRectangle(size=self.size,pos=self.pos,radius=[dp(14)])
        self.bind(pos=self._upd,size=self._upd)
        # Left color bar
        bar=Widget(size_hint=(None,1),width=dp(5),pos_hint={'x':0,'y':0})
        with bar.canvas:
            Color(*color)
            RoundedRectangle(pos=bar.pos,size=bar.size,radius=[dp(3)])
        bar.bind(pos=self._rb(color),size=self._rb(color))
        self.add_widget(bar)
        # Text
        txt=Label(text=f"[b][size=16sp]{title}[/size][/b]\n[size=11sp]{desc}[/size]",
                  markup=True,color=WHT,halign='left',valign='middle',
                  pos_hint={'x':.09,'center_y':.5},size_hint=(.75,1))
        self.add_widget(txt)
        # Arrow
        self.add_widget(Label(text=">",font_size=sp(22),color=TXT2,pos_hint={'right':.97,'center_y':.5},
                              size_hint=(None,1),width=dp(20),halign='right',valign='middle'))
        # Invisible Button overlay (covers entire card)
        if on_press:
            btn=Button(text=".",background_normal='',background_color=(0,0,0,0),color=(0,0,0,0),
                       pos_hint={'x':0,'y':0},size_hint=(1,1),font_size=sp(1))
            btn.bind(on_press=on_press)
            self.add_widget(btn)
    def _upd(self,*a): self._bg.size=self.size; self._bg.pos=self.pos
    def _rb(self,c):
        def _cb(w,v):
            w.canvas.clear()
            with w.canvas:
                Color(*c)
                RoundedRectangle(pos=w.pos,size=w.size,radius=[dp(3)])
        return _cb

# ============= Screen 1: Main =============
class MainScreen(Screen):
    def __init__(self,**kw):
        super().__init__(**kw)
        self.dm=DataManager()
        with self.canvas.before: Color(*BG); self._bg=Rectangle(size=self.size,pos=self.pos)
        self.bind(size=self._r,pos=self._r)
        root=BoxLayout(orientation='vertical',padding=dp(16),spacing=dp(10))

        # === Hero ===
        hero=FloatLayout(size_hint=(1,.24))
        with hero.canvas.before:
            Color(*get_color_from_hex("#130A0A"))
            RoundedRectangle(pos=hero.pos,size=hero.size,radius=[0,0,dp(32),dp(32)])
            # Accent line
            Color(*get_color_from_hex("#FF444433"))
            Rectangle(pos=(hero.x,hero.y),size=(hero.width,dp(2)))
        hero.bind(pos=lambda i,v:self._rh(i),size=lambda i,v:self._rh(i))
        # Decorative balls (dim, background)
        for num,x,y,sz,op in [(7,.18,.48,dp(36),.15),(16,.78,.35,dp(40),.12),(33,.55,.62,dp(28),.10)]:
            b=Ball(number=num,is_blue=(num==16),sz=sz)
            b.pos_hint={'center_x':x,'center_y':y}; b.opacity=op; hero.add_widget(b)
        # Title
        hero.add_widget(Label(text="[b]SSQ[/b] Predictor",markup=True,font_size=sp(30),
                              color=get_color_from_hex("#FF5555"),
                              pos_hint={'center_x':.5,'center_y':.72},
                              size_hint=(None,None),size=(dp(300),dp(44)),halign='center'))
        hero.add_widget(Label(text="双色球 · AI 预测系统",font_size=sp(13),color=TXT2,
                              pos_hint={'center_x':.5,'center_y':.38},
                              size_hint=(None,None),size=(dp(200),dp(22)),halign='center'))
        hero.add_widget(Label(text="MCMC + 自注意力 + 关联挖掘",font_size=sp(10),color=TXT2,
                              pos_hint={'center_x':.5,'center_y':.2},
                              size_hint=(None,None),size=(dp(240),dp(16)),halign='center'))
        root.add_widget(hero)

        # === Status Pill ===
        pill=FloatLayout(size_hint=(1,.09))
        with pill.canvas.before:
            Color(*SURF); RoundedRectangle(pos=pill.pos,size=pill.size,radius=[dp(14)])
        pill.bind(pos=lambda i,v:self._rd(i),size=lambda i,v:self._rd(i))
        # Left: data count
        self._info=Label(text="[b]加载中…[/b]",markup=True,font_size=sp(13),color=TXT,
                         pos_hint={'x':.05,'center_y':.5},size_hint=(.55,1),halign='left',valign='middle')
        pill.add_widget(self._info)
        # Right: status dot
        self._dot=Label(text="○",font_size=sp(16),color=TXT2,
                        pos_hint={'right':.96,'center_y':.5},size_hint=(None,1),width=dp(20),halign='right',valign='middle')
        pill.add_widget(self._dot)
        self._tag=Label(text="",font_size=sp(10),color=TXT2,
                        pos_hint={'right':.88,'center_y':.5},size_hint=(None,1),width=dp(56),halign='right',valign='middle')
        pill.add_widget(self._tag)
        root.add_widget(pill)

        # === Feature Cards ===
        cards=BoxLayout(orientation='vertical',spacing=dp(10),size_hint=(1,.50))
        cards.add_widget(CardBtn("刷新数据","联网抓取最新开奖结果",BLUE,on_press=self._refresh))
        cards.add_widget(CardBtn("历史开奖","查看最近 10 期开奖号码",GRN,on_press=lambda x:self._nav('history')))
        cards.add_widget(CardBtn("AI 预测","MCMC 采样 · 综合评分 · 权重热力图",RED,on_press=lambda x:self._nav('predict')))
        root.add_widget(cards)

        # === Progress ===
        self._prog=ProgressBar(max=100,value=0,size_hint=(1,.025)); self._prog.opacity=0
        root.add_widget(self._prog)

        # === Footer ===
        ft=Label(text="v1.0 · 更多数据 = 更准预测",font_size=sp(9),color=get_color_from_hex("#3A3F4A"),
                 size_hint=(1,.03),halign='center',valign='middle')
        root.add_widget(ft)

        self.add_widget(root)
        Clock.schedule_once(lambda dt:self._init(),.2)

    def _init(self):
        try:
            s=self.dm.load()
            self._info.text=f"[b]{s['总期数']:,}[/b] 期  |  {s['最新一期']}期  |  {s.get('日期范围','')[:4]}~{s.get('日期范围','')[-4:]}"
            self._dot.text="●"; self._tag.text="就绪"
        except:
            self._info.text="数据未找到"; self._dot.text="○"; self._tag.text=""

    def _refresh(self,*a):
        if self._tag.text=="联网中…": return  # prevent double-click
        self._tag.text="联网中…"; self._dot.text="◌"
        self._prog.opacity=1; self._prog.value=0
        def run():
            ok=self.dm.refresh()
            Clock.schedule_once(lambda dt:self._done(ok),0)
        threading.Thread(target=run,daemon=True).start()

    def _done(self,ok):
        try:
            self._prog.opacity=0
            if ok: self._tag.text="已更新"; self._dot.text="●"
            else: self._tag.text="失败"; self._dot.text="✕"
            self._init()
        except Exception as e:
            self._info.text=f"出错:{str(e)[:30]}"; self._dot.text="✕"

    def _nav(self,name):
        if name=='history': self.manager.get_screen('history').load(self.dm)
        elif name=='predict': self.manager.get_screen('predict').start(self.dm)
        self.manager.current=name

    def _r(self,*a): self._bg.size=self.size; self._bg.pos=self.pos
    def _rh(self,i,*a):
        for c in i.canvas.before.children:
            if isinstance(c,RoundedRectangle): c.size=i.size; c.pos=i.pos
    def _rd(self,i,*a):
        for c in i.canvas.before.children:
            if isinstance(c,RoundedRectangle): c.size=i.size; c.pos=i.pos

# ============= Screen 2: History =============
class HistoryScreen(Screen):
    def __init__(self,**kw):
        super().__init__(**kw)
        with self.canvas.before: Color(*BG); self._bg=Rectangle(size=self.size,pos=self.pos)
        self.bind(size=self._r,pos=self._r)
        root=BoxLayout(orientation='vertical',padding=dp(14),spacing=dp(8))
        hdr=BoxLayout(orientation='horizontal',size_hint=(1,.07))
        back=Button(text="< 返回",font_size=sp(14),background_color=(0,0,0,0),color=BLUE,size_hint=(.25,1),halign='left',valign='middle')
        back.bind(on_press=lambda x:setattr(self.manager,'current','main'))
        hdr.add_widget(back); hdr.add_widget(Label(text="开奖历史",font_size=sp(18),bold=True,color=TXT,halign='center',valign='middle')); hdr.add_widget(Widget(size_hint_x=.25))
        root.add_widget(hdr)
        hdr2=BoxLayout(orientation='horizontal',size_hint_y=None,height=dp(22),padding=dp(4))
        for lbl in ["期号","日期","红球","蓝球"]:
            hdr2.add_widget(Label(text=lbl,font_size=sp(10),color=TXT2,halign='left' if lbl=="期号" else 'center',valign='middle'))
        root.add_widget(hdr2)
        self.scroll=ScrollView(size_hint=(1,.88))
        self.content=GridLayout(cols=1,spacing=dp(6),size_hint_y=None,padding=dp(2))
        self.content.bind(minimum_height=self.content.setter('height'))
        self.scroll.add_widget(self.content)
        root.add_widget(self.scroll)
        self.add_widget(root)

    def load(self,dm):
        self.content.clear_widgets()
        for d in dm.get_last_n(10):
            reds=[int(x) for x in d["红球"].split()] if isinstance(d["红球"],str) else d["红球"]
            blue=int(d["蓝球"]) if isinstance(d["蓝球"],str) else d["蓝球"]
            # Card
            card=BoxLayout(orientation='horizontal',size_hint_y=None,height=dp(50),padding=dp(6),spacing=dp(4))
            def _bg(c,r):
                c.canvas.before.clear()
                with c.canvas.before:
                    Color(*SURF2 if int(d.get('期号','0')[-1])%2 else SURF)
                    RoundedRectangle(pos=c.pos,size=c.size,radius=[dp(10)])
            card.bind(pos=_bg,size=_bg)
            # Issue + Date
            info=Label(text=f"{d.get('期号','')}\n[size=10sp]{d.get('开奖日期',d.get('日期',''))[-5:]}[/size]",
                       markup=True,font_size=sp(12),bold=True,color=TXT,
                       size_hint=(.25,1),halign='left',valign='middle')
            card.add_widget(info)
            # Ball area - auto-distribute
            balls=BoxLayout(orientation='horizontal',spacing=dp(2),size_hint=(.75,1),padding=dp(2))
            # Use size_hint to auto-scale
            for r in reds:
                balls.add_widget(Ball(number=r,is_blue=False,sz=dp(28)))
            balls.add_widget(Widget(size_hint_x=None,width=dp(6)))
            balls.add_widget(Ball(number=blue,is_blue=True,sz=dp(30)))
            card.add_widget(balls)
            self.content.add_widget(card)

    def _r(self,*a): self._bg.size=self.size; self._bg.pos=self.pos
    def _rc(self,c,*a):
        for ch in c.canvas.before.children:
            if isinstance(ch,RoundedRectangle): ch.size=c.size; ch.pos=c.pos

# ============= Screen 3: Predict =============
class PredictScreen(Screen):
    def __init__(self,**kw):
        super().__init__(**kw)
        with self.canvas.before: Color(*BG); self._bg=Rectangle(size=self.size,pos=self.pos)
        self.bind(size=self._r,pos=self._r)
        root=BoxLayout(orientation='vertical',padding=dp(14),spacing=dp(8))
        # Header
        hdr=BoxLayout(orientation='horizontal',size_hint=(1,.07))
        back=Button(text="< 返回",font_size=sp(14),background_color=(0,0,0,0),color=BLUE,size_hint=(.25,1),halign='left',valign='middle')
        back.bind(on_press=lambda x:setattr(self.manager,'current','main'))
        hdr.add_widget(back); hdr.add_widget(Label(text="AI 预测",font_size=sp(18),bold=True,color=TXT,halign='center',valign='middle')); hdr.add_widget(Widget(size_hint_x=.25))
        root.add_widget(hdr)
        self._st=Label(text="点击下方按钮开始预测",font_size=sp(13),color=TXT2,size_hint=(1,.06),halign='center')
        root.add_widget(self._st)
        self._prog=ProgressBar(max=100,value=0,size_hint=(1,.02))
        root.add_widget(self._prog)
        # Scroll results
        self.scroll=ScrollView(size_hint=(1,.70))
        self._box=BoxLayout(orientation='vertical',size_hint_y=None,spacing=dp(8),padding=dp(4))
        self._box.bind(minimum_height=self._box.setter('height'))
        self.scroll.add_widget(self._box)
        root.add_widget(self.scroll)
        # Start button
        btn=Button(text="▶  开始预测",font_size=sp(16),bold=True,
                   background_normal='',background_color=RED,color=WHT,size_hint=(1,.08))
        btn.bind(on_press=self._start)
        root.add_widget(btn)
        self.add_widget(root)
        self._busy=False

    def start(self,dm): self._start()
    def _start(self,*a):
        if self._busy: return
        self._busy=True; self._box.clear_widgets(); self._st.text="准备模型…"; self._prog.value=0
        dm=self.manager.get_screen('main').dm
        def run():
            try:
                self._ui("加载数据…",5); data=dm.parsed if dm.parsed else load_parsed()
                self._ui("拟合分布…",15); pred=SSQPredictor(data,half_life=200); pred.mc_n_samples=3000
                pred.prepare(train_mcmc=True,mine_associations=True,min_support=0.01)
                self._ui("MCMC采样…",30); result=pred.predict(top_n=10,n_mc_samples=3000)
                self._ui("完成",95)
                Clock.schedule_once(lambda dt:self._show(result,pred),0)
            except Exception as e: Clock.schedule_once(lambda dt:self._err(str(e)),0)
            self._busy=False
        threading.Thread(target=run,daemon=True).start()

    def _ui(self,msg,p):
        try: Clock.schedule_once(lambda dt:(setattr(self._st,'text',msg),setattr(self._prog,'value',p)),0)
        except: pass
    def _err(self,msg):
        try: self._st.text=f"失败:{msg[:50]}"; self._prog.value=0
        except: pass
    def _r(self,*a):
        try: self._bg.size=self.size; self._bg.pos=self.pos
        except: pass

    def _show(self,result,pred):
        self._prog.value=100
        self._st.text=f"接受率{result['diagnostics']['acceptance_rate']:.2f} | 候选{result['n_candidates_generated']}组 | {pred.n_draws}期数据"
        self._box.clear_widgets()

        # ---- Stats Card ----
        stat=FloatLayout(size_hint_y=None,height=dp(48))
        with stat.canvas.before:
            Color(*SURF); RoundedRectangle(pos=stat.pos,size=stat.size,radius=[dp(12)])
        stat.bind(pos=lambda i,v:self._rs(i),size=lambda i,v:self._rs(i))
        diag=result['diagnostics']
        stat.add_widget(Label(text=f"接受率\n[color=#44FF88]{diag['acceptance_rate']:.3f}[/color]",
                              markup=True,font_size=sp(11),color=TXT2,
                              pos_hint={'x':.03,'center_y':.5},size_hint=(.3,1),halign='center',valign='middle'))
        stat.add_widget(Label(text=f"候选组\n[color=#FF8844]{result['n_candidates_generated']}[/color]",
                              markup=True,font_size=sp(11),color=TXT2,
                              pos_hint={'x':.35,'center_y':.5},size_hint=(.3,1),halign='center',valign='middle'))
        stat.add_widget(Label(text=f"历史数据\n[color=#4488FF]{pred.n_draws}期[/color]",
                              markup=True,font_size=sp(11),color=TXT2,
                              pos_hint={'x':.67,'center_y':.5},size_hint=(.3,1),halign='center',valign='middle'))
        self._box.add_widget(stat)
        self._box.add_widget(Widget(size_hint_y=None,height=dp(8)))

        # ---- Recommendations ----
        self._box.add_widget(Label(text="[b]推荐方案 Top-10[/b]",markup=True,font_size=sp(16),color=GOLD,
                                   size_hint_y=None,height=dp(32),halign='left'))
        for i,rec in enumerate(result["recommendations"]):
            # Use BoxLayout row — auto-prevents overlap
            row=BoxLayout(orientation='horizontal',size_hint_y=None,height=dp(42),spacing=dp(2),padding=dp(4))
            bg_c=GOLD if i==0 else (SURF2 if i<3 else SURF)
            with row.canvas.before: Color(*bg_c); RoundedRectangle(pos=row.pos,size=row.size,radius=[dp(12)])
            row.bind(pos=lambda i,v:self._rs(i),size=lambda i,v:self._rs(i))
            # Rank
            bc=GOLD if i==0 else RED if i==1 else BLUE if i==2 else TXT2
            rank=Label(text=str(i+1),font_size=sp(12),bold=True,color=(0,0,0,1) if i==0 else WHT,
                       size_hint=(None,1),width=dp(22),halign='center',valign='middle')
            with rank.canvas.before: Color(*bc); RoundedRectangle(pos=rank.pos,size=rank.size,radius=[dp(11)])
            rank.bind(pos=lambda w,v,b=bc:self._bd(w,b),size=lambda w,v,b=bc:self._bd(w,b))
            row.add_widget(rank)
            # Red balls - take remaining space
            red_box=BoxLayout(orientation='horizontal',spacing=dp(1),size_hint=(.55,1))
            for r in rec["红球"]: red_box.add_widget(Ball(number=r,is_blue=False,sz=dp(26)))
            row.add_widget(red_box)
            # Spacer
            row.add_widget(Widget(size_hint_x=.02))
            # Blue
            row.add_widget(Ball(number=rec["蓝球"],is_blue=True,sz=dp(28)))
            # Score
            row.add_widget(Widget(size_hint_x=.02))
            row.add_widget(Label(text=f"{rec['combined_score']:.1f}",font_size=sp(10),color=TXT2,
                                 size_hint=(None,1),width=dp(36),halign='right',valign='middle'))
            self._box.add_widget(row)

        # ---- Red Heatmap ----
        self._box.add_widget(Widget(size_hint_y=None,height=dp(16)))
        self._box.add_widget(Label(text="[b]红球权重[/b]  越红越热 · 越大越推荐",markup=True,font_size=sp(14),color=RED,
                                   size_hint_y=None,height=dp(26),halign='left'))
        rp=result["marginal_probs"]["red"]; rs=sorted(rp.items(),key=lambda x:x[1],reverse=True)
        mx=rs[0][1] if rs else 1.0
        # Responsive grid: 11 cols on wide, 6 cols on narrow
        w=Window.width; ncols=11 if w>dp(400) else 6
        cell_w=(w-dp(20))/ncols-dp(2)
        rows=math.ceil(33/ncols)
        grid=GridLayout(cols=ncols,spacing=dp(2),size_hint_y=None,height=dp(40)*rows,padding=dp(2))
        for b,p in rs:
            t=p/max(mx,.001)
            lo=get_color_from_hex("#1E2228"); hi=RED
            bg=tuple(lo[i]+(hi[i]-lo[i])*(t**.4) for i in range(4))
            cell=Label(text=f"{str(b).zfill(2)}\n{p*100:.1f}%",font_size=sp(9+4*t),bold=t>.55,
                       color=WHT,size_hint=(None,None),width=dp(max(38,cell_w/dp(1))),height=dp(38),
                       halign='center',valign='middle')
            with cell.canvas.before: Color(*bg); RoundedRectangle(pos=cell.pos,size=cell.size,radius=[dp(8)])
            cell.bind(pos=lambda w,v,bg=bg:self._ht(w,bg),size=lambda w,v,bg=bg:self._ht(w,bg))
            grid.add_widget(cell)
        self._box.add_widget(grid)

        # ---- Blue Weights ----
        self._box.add_widget(Widget(size_hint_y=None,height=dp(12)))
        self._box.add_widget(Label(text="[b]蓝球权重[/b]  越蓝越热 · 越大越推荐",markup=True,font_size=sp(14),color=BLUE,
                                   size_hint_y=None,height=dp(26),halign='left'))
        bp=result["marginal_probs"]["blue"]; bs=sorted(bp.items(),key=lambda x:x[1],reverse=True)
        mxb=bs[0][1] if bs else 1.0
        bncols=8 if Window.width>dp(400) else 4
        bcell_w=(Window.width-dp(20))/bncols-dp(2)
        g2=GridLayout(cols=bncols,spacing=dp(2),size_hint_y=None,height=dp(36)*math.ceil(16/bncols),padding=dp(2))
        for b,p in bs:
            t=p/max(mxb,.001)
            lo=get_color_from_hex("#1E2228"); hi=BLUE
            bg=tuple(lo[i]+(hi[i]-lo[i])*(t**.4) for i in range(4))
            cell=Label(text=f"{str(b).zfill(2)}\n{p*100:.1f}%",font_size=sp(9+3*t),bold=t>.55,
                       color=WHT,size_hint=(None,None),width=dp(max(34,bcell_w/dp(1))),height=dp(36),
                       halign='center',valign='middle')
            with cell.canvas.before: Color(*bg); RoundedRectangle(pos=cell.pos,size=cell.size,radius=[dp(6)])
            cell.bind(pos=lambda w,v,bg=bg:self._ht(w,bg),size=lambda w,v,bg=bg:self._ht(w,bg))
            g2.add_widget(cell)
        self._box.add_widget(g2)
        self._box.add_widget(Widget(size_hint_y=None,height=dp(8)))

    def _bd(self,w,c):
        w.canvas.before.clear()
        with w.canvas.before:
            Color(*c)
            RoundedRectangle(pos=w.pos,size=w.size,radius=[dp(12)])
    def _ht(self,w,bg):
        w.canvas.before.clear()
        with w.canvas.before:
            Color(*bg)
            RoundedRectangle(pos=w.pos,size=w.size,radius=[dp(8)])
    def _rs(self,i,*a):
        for c in i.canvas.before.children:
            if isinstance(c,RoundedRectangle): c.size=i.size; c.pos=i.pos

# ============= App =============
class SSQApp(App):
    title="SSQ Predictor"
    def build(self):
        Window.clearcolor=BG
        sm=ScreenManager(transition=SlideTransition(duration=.2))
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(HistoryScreen(name='history'))
        sm.add_widget(PredictScreen(name='predict'))
        return sm

if __name__=='__main__': SSQApp().run()
