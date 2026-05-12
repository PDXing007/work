#!/usr/bin/env python3
"""
SSQ Predictor — Kivy 移动端应用 (美化版)

Material Design 风格，三屏结构：
  - 主页：顶部 Hero + 数据仪表盘 + 三个功能卡片
  - 历史：彩票票根式卡片，渐变球体
  - 预测：进度动画 + 金牌推荐 + 色温权重矩阵
"""

import os, sys, json, math, threading, random
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------- Kivy ----------
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition, RiseInTransition
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.graphics import (
    Color, Rectangle, RoundedRectangle, Ellipse, Line, Mesh, PushMatrix, PopMatrix, Rotate,
)
from kivy.graphics.context_instructions import Transform
from kivy.animation import Animation
from kivy.utils import get_color_from_hex, platform
from kivy.properties import NumericProperty, StringProperty, ListProperty, BooleanProperty

# ---------- 预测 & 爬虫 ----------
from data_loader import load_history, load_parsed, get_data_summary
from predictor import SSQPredictor
from features_basic import compute_global_frequencies
from fetcher import (
    load_existing, save_data, parse_issue, HEADERS,
)

# ====================================================================
# 调色板 — Material Dark + Lottery Accent
# ====================================================================
C_BG        = get_color_from_hex("#0D1117")      # 深色背景
C_SURFACE   = get_color_from_hex("#161B22")      # 卡片面
C_SURFACE2  = get_color_from_hex("#1C2333")      # 次级面
C_BORDER    = get_color_from_hex("#30363D")      # 边框
C_PRIMARY   = get_color_from_hex("#E84040")      # 主色-红
C_ACCENT    = get_color_from_hex("#3B82F6")      # 强调-蓝
C_GOLD      = get_color_from_hex("#F0B90B")      # 金牌色
C_GREEN     = get_color_from_hex("#22C55E")      # 成功绿
C_TEXT      = get_color_from_hex("#E6EDF3")      # 主文字
C_TEXT2     = get_color_from_hex("#8B949E")      # 次文字
C_TEXT3     = get_color_from_hex("#484F58")      # 弱文字
C_RED_GRAD  = [get_color_from_hex("#FF4444"), get_color_from_hex("#CC1111")]  # 红球渐变
C_BLUE_GRAD = [get_color_from_hex("#4488FF"), get_color_from_hex("#1155CC")]  # 蓝球渐变
C_WHITE     = (1, 1, 1, 1)


def _lighten(c, f=0.3):
    return tuple(min(1, x + f) for x in c[:3]) + (c[3],)


def _darken(c, f=0.2):
    return tuple(max(0, x - f) for x in c[:3]) + (c[3],)


def _lerp_color(c1, c2, t):
    return tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(4))


# ====================================================================
# 可复用部件
# ====================================================================

class RoundedButton(Button):
    """圆角按钮 — 带按下缩放反馈"""
    bg = ListProperty([0.2, 0.2, 0.3, 1])
    radius = NumericProperty(dp(14))

    def __init__(self, bg_color=None, **kw):
        super().__init__(**kw)
        self.background_normal = ''
        self.background_color = (0, 0, 0, 0)
        if bg_color:
            self.bg = bg_color
        self.bind(pos=self._draw, size=self._draw)

    def _draw(self, *a):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self.bg)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[self.radius])
            # 顶部高光线
            Color(1, 1, 1, 0.06)
            RoundedRectangle(
                pos=(self.x + dp(2), self.y + self.height * 0.7),
                size=(self.width - dp(4), self.height * 0.28),
                radius=[self.radius, self.radius, 0, 0],
            )

    def on_press(self):
        Animation(size=(self.width * 0.96, self.height * 0.96),
                  pos=(self.x + self.width * 0.02, self.y + self.height * 0.02),
                  duration=0.08).start(self)

    def on_release(self):
        Animation(size=(self.width * 1.0, self.height * 1.0),
                  pos=(self.x, self.y), duration=0.08).start(self)


class ShadowCard(FloatLayout):
    """带阴影的卡片容器"""
    def __init__(self, radius=dp(16), **kw):
        super().__init__(**kw)
        self._radius = radius
        self.bind(pos=self._draw, size=self._draw)

    def _draw(self, *a):
        self.canvas.before.clear()
        r = self._radius
        # 阴影层
        for i, (off, alpha) in enumerate([(dp(4), 0.15), (dp(2), 0.08), (dp(1), 0.04)]):
            with self.canvas.before:
                Color(0, 0, 0, alpha)
                RoundedRectangle(
                    pos=(self.x + off, self.y - off),
                    size=self.size, radius=[r]
                )
        # 卡片面
        with self.canvas.before:
            Color(*C_SURFACE)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[r])
            # 边框
            Color(*C_BORDER)
            Line(rounded_rectangle=(*self.pos, *self.size, r), width=dp(0.8))


class BallWidget(Widget):
    """彩票球 — 带径向渐变模拟"""
    number = NumericProperty(0)
    is_blue = BooleanProperty(False)
    ball_size = NumericProperty(dp(36))

    def __init__(self, **kw):
        super().__init__(**kw)
        self.size_hint = (None, None)
        self.bind(pos=self._draw, size=self._draw, number=self._draw, is_blue=self._draw)

    def _draw(self, *a):
        self.canvas.clear()
        sz = self.ball_size
        cx, cy = self.x + sz / 2, self.y + sz / 2
        r = sz / 2 - dp(1)
        colors = C_BLUE_GRAD if self.is_blue else C_RED_GRAD

        with self.canvas:
            # 主圆
            Color(*colors[0])
            Ellipse(pos=(cx - r, cy - r), size=(r * 2, r * 2))
            # 高光
            Color(1, 1, 1, 0.25)
            Ellipse(pos=(cx - r * 0.55, cy - r * 0.05), size=(r * 1.1, r * 0.7))
            # 次高光
            Color(1, 1, 1, 0.10)
            Ellipse(pos=(cx - r * 0.2, cy + r * 0.15), size=(r * 0.4, r * 0.25))

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            Animation(ball_size=self.ball_size * 1.15, duration=0.1).start(self)
    def on_touch_up(self, touch):
        Animation(ball_size=dp(36), duration=0.1).start(self)


class LoadingSpinner(Widget):
    """旋转加载动画"""
    angle = NumericProperty(0)

    def __init__(self, **kw):
        super().__init__(**kw)
        self._anim = Animation(angle=360, duration=1.2) + Animation(angle=360, duration=1.2)
        self._anim.repeat = True
        self.bind(pos=self._draw, size=self._draw, angle=self._draw)

    def start(self):
        self._anim.start(self)

    def stop(self):
        self._anim.stop(self)

    def _draw(self, *a):
        self.canvas.clear()
        cx, cy = self.center_x, self.center_y
        r = min(self.width, self.height) / 2 - dp(4)
        with self.canvas:
            PushMatrix()
            Rotate(angle=self.angle, origin=(cx, cy))
            for i in range(8):
                alpha = 0.15 + 0.85 * (i / 7)
                Color(1, 1, 1, alpha)
                angle_rad = math.radians(i * 45)
                bx = cx + math.cos(angle_rad) * r * 0.6
                by = cy + math.sin(angle_rad) * r * 0.6
                Ellipse(pos=(bx - dp(3), by - dp(3)), size=(dp(6), dp(6)))
            PopMatrix()


# ====================================================================
# DataManager
# ====================================================================

class DataManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = None; self.parsed = None; self.summary = None
        self.status_msg = ""

    def load(self):
        with self.lock:
            self.data = load_history()
            self.parsed = load_parsed()
            self.summary = get_data_summary(self.data)
            return self.summary

    def get_last_n(self, n=10):
        with self.lock:
            if not self.parsed: self.load()
            return self.parsed[:n]

    def refresh(self):
        try:
            self.status_msg = "检查开奖数据..."
            data = load_existing()
            import requests
            s = requests.Session(); s.trust_env = True
            p = {"transactionType": "10001003", "lotteryId": "1", "count": "10"}
            r = s.get("https://jc.zhcw.com/port/client_json.php",
                      params=p, headers=HEADERS, timeout=15)
            api = r.json()
            if api.get("resCode") != "000000":
                self.status_msg = f"API: {api.get('message','?')}"; return False
            missing = [i for i in api["issue"] if i not in {d["期号"] for d in data}]
            if not missing:
                self.status_msg = f"已是最新 ({len(data)}期)"; return True
            self.status_msg = f"抓取中... 0/{len(missing)}"
            missing.reverse()
            for idx, iss in enumerate(missing):
                self.status_msg = f"抓取中... {idx+1}/{len(missing)}"
                p2 = {"transactionType": "10001002", "lotteryId": "1", "issue": iss}
                r2 = s.get("https://jc.zhcw.com/port/client_json.php",
                           params=p2, headers=HEADERS, timeout=15)
                if r2.status_code == 200:
                    d = parse_issue(r2.json())
                    if d: data.append(d)
            save_data(data); self.load()
            self.status_msg = f"已更新 {len(data)}期 (+{len(missing)})"
            return True
        except Exception as e:
            self.status_msg = f"网络错误: {str(e)[:40]}"; return False


# ====================================================================
# Screen 1: 主页
# ====================================================================

class MainScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.dm = DataManager()
        with self.canvas.before:
            Color(*C_BG); self._bg = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._r, pos=self._r)

        root = FloatLayout()

        # -------- 顶部 Hero --------
        hero = FloatLayout(size_hint=(1, 0.32), pos_hint={'top': 1})
        with hero.canvas.before:
            Color(*get_color_from_hex("#1A0A0A"))
            RoundedRectangle(pos=hero.pos, size=hero.size, radius=[0, 0, dp(32), dp(32)])
        hero.bind(pos=self._rd_hero, size=self._rd_hero)

        # Logo区：两个装饰球
        for i, (num, is_blue, ox, oy) in enumerate([
            (7, False, 0.25, 0.55), (12, True, 0.7, 0.35),
        ]):
            bw = BallWidget(number=num, is_blue=is_blue,
                            ball_size=dp(44), opacity=0.25,
                            pos_hint={'center_x': ox, 'center_y': oy})
            hero.add_widget(bw)

        hero_title = Label(
            text="SSQ\nPredictor",
            font_size=sp(30), bold=True,
            color=get_color_from_hex("#FF6666"),
            pos_hint={'center_x': 0.5, 'center_y': 0.65},
            halign='center', size_hint=(None, None),
            size=(dp(240), dp(70)),
        )
        hero.add_widget(hero_title)
        hero_sub = Label(
            text="双色球 · AI预测",
            font_size=sp(13), color=C_TEXT2,
            pos_hint={'center_x': 0.5, 'center_y': 0.35},
            halign='center', size_hint=(None, None),
            size=(dp(200), dp(24)),
        )
        hero.add_widget(hero_sub)
        root.add_widget(hero)

        # -------- 数据仪表盘 --------
        dash = ShadowCard(radius=dp(16),
                          size_hint=(0.9, 0.1),
                          pos_hint={'center_x': 0.5, 'top': 0.66})
        dash_content = BoxLayout(orientation='horizontal', padding=dp(12))
        self.dash_label = Label(
            text="加载中…",
            font_size=sp(13), color=C_TEXT,
            halign='left', valign='middle',
            size_hint_x=0.65,
        )
        dash_content.add_widget(self.dash_label)
        self.dash_badge = Label(
            text="", font_size=sp(11), color=C_GREEN,
            halign='right', valign='middle',
            size_hint_x=0.35, markup=True,
        )
        dash_content.add_widget(self.dash_badge)
        dash.add_widget(dash_content)
        root.add_widget(dash)

        # -------- 三个功能卡片 --------
        cards_data = [
            ("sync",    "刷新数据", "联网抓取最新\n开奖结果",     C_ACCENT,  self._do_refresh),
            ("history", "历史结果", "查看最近 10 期\n开奖号码",   C_GREEN,   self._go_history),
            ("predict", "开始预测", "MCMC 模型预测\n复式彩票权重", C_PRIMARY, self._go_predict),
        ]
        card_y = 0.52
        for icon, title, desc, color, cb in cards_data:
            card = self._feature_card(icon, title, desc, color, cb,
                                      pos_hint={'center_x': 0.5, 'y': card_y},
                                      size_hint=(0.9, 0.13))
            root.add_widget(card)
            card_y -= 0.145

        # -------- 底部状态 --------
        self.status_bar = Label(
            text="", font_size=sp(10), color=C_TEXT3,
            pos_hint={'center_x': 0.5, 'y': 0.01},
            size_hint=(0.9, 0.03), halign='center',
        )
        root.add_widget(self.status_bar)

        self.add_widget(root)
        Clock.schedule_once(lambda dt: self._init_load(), 0.3)

    def _feature_card(self, icon, title, desc, color, cb, **kw):
        card = ShadowCard(radius=dp(14), **kw)
        box = BoxLayout(orientation='horizontal', padding=dp(12), spacing=dp(10))
        # 左侧色条
        bar = Widget(size_hint=(None, 1), width=dp(5))
        with bar.canvas:
            Color(*color)
            RoundedRectangle(pos=bar.pos, size=bar.size, radius=[dp(3)])
        bar.bind(pos=lambda i,v: i.canvas.clear() or _draw_bar(i),
                 size=lambda i,v: i.canvas.clear() or _draw_bar(i))
        def _draw_bar(w):
            with w.canvas:
                Color(*color)
                RoundedRectangle(pos=w.pos, size=w.size, radius=[dp(3)])
        box.add_widget(bar)
        # 文字
        txt = BoxLayout(orientation='vertical', size_hint_x=0.7)
        txt.add_widget(Label(text=f"[b]{title}[/b]", markup=True,
                              font_size=sp(14), color=C_TEXT,
                              halign='left', valign='bottom',
                              size_hint_y=0.5))
        txt.add_widget(Label(text=desc, font_size=sp(11), color=C_TEXT2,
                              halign='left', valign='top',
                              size_hint_y=0.5))
        box.add_widget(txt)
        # 箭头
        arr = Label(text=">", font_size=sp(22), color=C_TEXT3,
                    size_hint_x=0.15, halign='right', valign='middle')
        box.add_widget(arr)
        card.add_widget(box)
        # 点击区域
        btn = Button(background_normal='', background_color=(0,0,0,0),
                     pos_hint={'x':0,'y':0}, size_hint=(1,1))
        btn.bind(on_press=cb)
        card.add_widget(btn)
        return card

    def _r(self, *a): self._bg.size = self.size; self._bg.pos = self.pos
    def _rd_hero(self, h, v):
        for c in h.canvas.before.children:
            if isinstance(c, RoundedRectangle) and c.radius == [0,0,dp(32),dp(32)]:
                c.size = h.size; c.pos = h.pos

    def _init_load(self):
        try:
            s = self.dm.load()
            self.dash_label.text = (
                f"[b]{s['总期数']:,}[/b] 期数据\n"
                f"第{s['最新一期']}期 · {s['日期范围'][:4]}~{s['日期范围'][-4:]}"
            )
            self.dash_label.markup = True
            self.dash_badge.text = "[b]●[/b] 就绪"
            self.status_bar.text = f"最后更新: {datetime.now().strftime('%H:%M')}"
        except Exception as e:
            self.dash_label.text = f"加载失败\n{str(e)[:30]}"

    def _do_refresh(self, *a):
        self.dash_badge.text = "[b]◌[/b] 刷新中"
        self.status_bar.text = "联网中…"
        def run():
            ok = self.dm.refresh()
            Clock.schedule_once(lambda dt: self._refresh_done(ok), 0)
        threading.Thread(target=run, daemon=True).start()

    def _refresh_done(self, ok):
        if ok:
            self._init_load()
            self.dash_badge.text = "[b]●[/b] 已更新"
        else:
            self.dash_badge.text = "[b]![/b] 失败"
        self.status_bar.text = self.dm.status_msg

    def _go_history(self, *a):
        self.manager.get_screen('history').load_data(self.dm)
        self.manager.current = 'history'
    def _go_predict(self, *a):
        self.manager.get_screen('predict').start_prediction(self.dm)
        self.manager.current = 'predict'


# ====================================================================
# Screen 2: 历史开奖
# ====================================================================

class HistoryScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        with self.canvas.before:
            Color(*C_BG); self._bg = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._r, pos=self._r)

        root = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(10))

        # 顶栏
        hdr = BoxLayout(orientation='horizontal', size_hint=(1, 0.07), spacing=dp(8))
        back = Button(text="←  返回", font_size=sp(13),
                      background_color=(0,0,0,0), color=C_ACCENT,
                      size_hint=(0.28, 1), halign='left', valign='middle')
        back.bind(on_press=lambda x: setattr(self.manager, 'current', 'main'))
        hdr.add_widget(back)
        hdr.add_widget(Label(text="[b]开奖历史[/b]", markup=True,
                              font_size=sp(17), color=C_TEXT,
                              halign='center', valign='middle'))
        # 占位
        hdr.add_widget(Widget(size_hint_x=0.28))
        root.add_widget(hdr)

        # 线段
        sep = Widget(size_hint=(1, None), height=dp(1))
        with sep.canvas:
            Color(*C_BORDER); Rectangle(size=sep.size, pos=sep.pos)
        sep.bind(pos=lambda i,v: _sep(i), size=lambda i,v: _sep(i))
        def _sep(w):
            w.canvas.clear()
            with w.canvas:
                Color(*C_BORDER)
                Rectangle(size=w.size, pos=w.pos)
        root.add_widget(sep)

        self.scroll = ScrollView(size_hint=(1, 0.93))
        self.content = GridLayout(cols=1, spacing=dp(12),
                                  size_hint_y=None, padding=dp(4))
        self.content.bind(minimum_height=self.content.setter('height'))
        self.scroll.add_widget(self.content)
        root.add_widget(self.scroll)
        self.add_widget(root)

    def load_data(self, dm):
        self.content.clear_widgets()
        for d in dm.get_last_n(10):
            self.content.add_widget(self._ticket(d))

    def _ticket(self, d):
        # 票根卡片
        card = FloatLayout(size_hint_y=None, height=dp(130))
        with card.canvas.before:
            Color(*C_SURFACE)
            RoundedRectangle(pos=card.pos, size=card.size, radius=[dp(14)])
            Color(*C_BORDER)
            Line(rounded_rectangle=(*card.pos, *card.size, dp(14)), width=dp(0.7))
        card.bind(pos=lambda i,v: self._rc(card), size=lambda i,v: self._rc(card))

        # 锯齿边效果 — 左右两侧的半圆
        for side_x_factor in [0, 1]:
            for j in range(5):
                dot = Widget(pos=(0, 0), size=(dp(12), dp(12)))
                with dot.canvas:
                    Color(*C_BG)
                    Ellipse(pos=dot.pos, size=dot.size)
                _j = j; _fx = side_x_factor
                def _make_dot_cb(card, row_j, row_fx):
                    def _cb(inst, val):
                        inst.canvas.clear()
                        with inst.canvas:
                            Color(*C_BG)
                            cx = card.x - dp(6) + row_fx * (card.width)
                            cy = card.y + dp(18) + row_j * dp(22)
                            inst.pos = (cx, cy)
                            Ellipse(pos=inst.pos, size=inst.size)
                    return _cb
                dot.bind(pos=_make_dot_cb(card, _j, _fx))
                card.add_widget(dot)

        # 期号
        issue_lbl = Label(
            text=f"第 {d['期号']} 期",
            font_size=sp(14), bold=True, color=C_TEXT,
            pos_hint={'x': 0.06, 'top': 0.92},
            size_hint=(0.5, None), height=dp(22), halign='left',
        )
        date = d.get('开奖日期', d.get('日期', ''))
        date_lbl = Label(
            text=date, font_size=sp(11), color=C_TEXT2,
            pos_hint={'right': 0.94, 'top': 0.92},
            size_hint=(0.4, None), height=dp(22), halign='right',
        )
        card.add_widget(issue_lbl); card.add_widget(date_lbl)

        # 红球
        reds = [int(x) for x in d["红球"].split()] if isinstance(d["红球"], str) else d["红球"]
        for i, r in enumerate(reds):
            bw = BallWidget(number=r, is_blue=False, ball_size=dp(38),
                            pos_hint={'x': 0.04 + i * 0.145, 'top': 0.68})
            card.add_widget(bw)

        # 蓝球
        blue = int(d["蓝球"]) if isinstance(d["蓝球"], str) else d["蓝球"]
        bw_b = BallWidget(number=blue, is_blue=True, ball_size=dp(42),
                          pos_hint={'right': 0.92, 'top': 0.68})
        card.add_widget(bw_b)
        bl = Label(text="蓝球", font_size=sp(10), color=C_TEXT3,
                   pos_hint={'right': 0.92, 'top': 0.38},
                   size_hint=(None, None), size=(dp(40), dp(16)), halign='center')
        card.add_widget(bl)

        # 销售额
        sales = d.get('销售额', 0)
        if sales and int(sales) > 0:
            s_lbl = Label(
                text=f"销量 {int(sales)//10000:,} 万",
                font_size=sp(10), color=C_TEXT3,
                pos_hint={'x': 0.06, 'y': 0.05},
                size_hint=(0.5, None), height=dp(16), halign='left',
            )
            card.add_widget(s_lbl)

        return card

    def _r(self, *a): self._bg.size = self.size; self._bg.pos = self.pos
    def _rc(self, card):
        for c in card.canvas.before.children:
            if isinstance(c, (RoundedRectangle, Rectangle)):
                c.size = card.size; c.pos = card.pos


# ====================================================================
# Screen 3: 预测结果
# ====================================================================

class PredictScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        with self.canvas.before:
            Color(*C_BG); self._bg = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._r, pos=self._r)

        root = BoxLayout(orientation='vertical', padding=dp(12), spacing=dp(8))

        # 顶栏
        hdr = BoxLayout(orientation='horizontal', size_hint=(1, 0.07), spacing=dp(8))
        back = Button(text="←  返回", font_size=sp(13),
                      background_color=(0,0,0,0), color=C_ACCENT,
                      size_hint=(0.28, 1), halign='left', valign='middle')
        back.bind(on_press=lambda x: setattr(self.manager, 'current', 'main'))
        hdr.add_widget(back)
        self.pred_title = Label(text="[b]AI 预测[/b]", markup=True,
                                font_size=sp(17), color=C_TEXT,
                                halign='center', valign='middle')
        hdr.add_widget(self.pred_title)
        hdr.add_widget(Widget(size_hint_x=0.28))
        root.add_widget(hdr)

        # 进度区
        self.prog_area = FloatLayout(size_hint=(1, 0.13))
        self.spinner = LoadingSpinner(
            pos_hint={'center_x': 0.5, 'center_y': 0.55},
            size_hint=(None, None), size=(dp(36), dp(36)),
        )
        self.prog_area.add_widget(self.spinner)
        self.prog_label = Label(
            text="点击下方按钮开始预测",
            font_size=sp(13), color=C_TEXT2,
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            halign='center',
        )
        self.prog_area.add_widget(self.prog_label)
        self.prog_bar = ProgressBar(max=100, value=0,
                                    size_hint=(0.8, None), height=dp(3),
                                    pos_hint={'center_x': 0.5, 'y': 0.1})
        self.prog_area.add_widget(self.prog_bar)
        root.add_widget(self.prog_area)

        # 分隔
        sep = Widget(size_hint=(1, None), height=dp(1))
        with sep.canvas:
            Color(*C_BORDER); Rectangle(size=sep.size, pos=sep.pos)
        sep.bind(pos=lambda i,v: _s(i), size=lambda i,v: _s(i))
        def _s(w):
            w.canvas.clear()
            with w.canvas:
                Color(*C_BORDER)
                Rectangle(size=w.size, pos=w.pos)
        root.add_widget(sep)

        # 结果滚动区
        self.scroll = ScrollView(size_hint=(1, 0.72))
        self.result_box = BoxLayout(orientation='vertical',
                                    size_hint_y=None, spacing=dp(10), padding=dp(4))
        self.result_box.bind(minimum_height=self.result_box.setter('height'))
        self.scroll.add_widget(self.result_box)
        root.add_widget(self.scroll)

        # 开始按钮
        self.start_btn = RoundedButton(
            bg_color=C_PRIMARY,
            text="[b]▶  开始预测[/b]", markup=True,
            font_size=sp(15), color=C_WHITE,
            size_hint=(1, 0.08),
        )
        self.start_btn.bind(on_press=self._start)
        root.add_widget(self.start_btn)

        self.add_widget(root)

    def _start(self, *a):
        self.start_btn.disabled = True
        self.result_box.clear_widgets()
        self.spinner.start()
        self.prog_label.text = "加载数据…"
        self.prog_bar.value = 0
        self.pred_title.text = "[b]预测中…[/b]"

        dm = self.manager.get_screen('main').dm
        self._thread_run(dm)

    def start_prediction(self, dm):
        """从MainScreen直接跳转时调用"""
        self._start()

    def _thread_run(self, dm):
        def run():
            try:
                self._ui("分析特征分布…", 10)
                data = dm.parsed if dm.parsed else load_parsed()

                self._ui("拟合统计模型…", 20)
                pred = SSQPredictor(data, half_life=200)
                pred.mc_n_samples = 5000
                pred.prepare(train_mcmc=True, mine_associations=True, min_support=0.01)

                self._ui("MCMC 采样…", 40)
                result = pred.predict(top_n=10, n_mc_samples=5000)

                self._ui("生成报告…", 85)
                Clock.schedule_once(lambda dt: self._show(result, pred), 0)
            except Exception as e:
                Clock.schedule_once(lambda dt: self._err(str(e)), 0)
        threading.Thread(target=run, daemon=True).start()

    def _ui(self, msg, prog):
        Clock.schedule_once(lambda dt: (
            setattr(self.prog_label, 'text', msg),
            setattr(self.prog_bar, 'value', prog),
        ), 0)

    def _err(self, msg):
        self.spinner.stop()
        self.start_btn.disabled = False
        self.pred_title.text = "[b]预测失败[/b]"
        self.prog_label.text = f"错误: {msg[:50]}"
        self.prog_bar.value = 0

    def _show(self, result, pred):
        self.spinner.stop()
        self.start_btn.disabled = False
        self.pred_title.text = "[b]预测完成[/b]"
        self.prog_label.text = (
            f"接受率 {result['diagnostics']['acceptance_rate']:.2f}  ·  "
            f"{result['n_candidates_generated']} 候选  ·  "
            f"{pred.n_draws} 期数据"
        )
        self.prog_bar.value = 100
        self.result_box.clear_widgets()

        # ---------- Section A: Top-10 方案 ----------
        self.result_box.add_widget(Label(
            text="[b]═══  推荐方案 Top-10  ═══[/b]",
            markup=True, font_size=sp(14), color=C_GOLD,
            size_hint_y=None, height=dp(32), halign='center',
        ))

        for i, rec in enumerate(result["recommendations"]):
            self.result_box.add_widget(self._rec_row(i + 1, rec))

        # ---------- Section B: 红球权重矩阵 ----------
        self.result_box.add_widget(Widget(size_hint_y=None, height=dp(12)))
        self.result_box.add_widget(Label(
            text="[b]═══  红球权重热力图  ═══[/b]",
            markup=True, font_size=sp(14), color=C_PRIMARY,
            size_hint_y=None, height=dp(32), halign='center',
        ))
        self.result_box.add_widget(Label(
            text="颜色越深 → 概率越高    字号越大 → 越推荐",
            font_size=sp(10), color=C_TEXT3,
            size_hint_y=None, height=dp(18), halign='center',
        ))

        red_probs = result["marginal_probs"]["red"]
        red_sorted = sorted(red_probs.items(), key=lambda x: x[1], reverse=True)
        max_rp = red_sorted[0][1] if red_sorted else 1.0

        heat_grid = GridLayout(cols=11, spacing=dp(3),
                               size_hint_y=None, height=dp(36) * 3, padding=dp(2))
        for ball, prob in red_sorted:
            heat_grid.add_widget(self._heat_cell(ball, prob, max_rp, True))
        self.result_box.add_widget(heat_grid)

        # ---------- Section C: 蓝球权重 ----------
        self.result_box.add_widget(Widget(size_hint_y=None, height=dp(8)))
        self.result_box.add_widget(Label(
            text="[b]═══  蓝球权重排名  ═══[/b]",
            markup=True, font_size=sp(14), color=C_ACCENT,
            size_hint_y=None, height=dp(32), halign='center',
        ))

        blue_probs = result["marginal_probs"]["blue"]
        blue_sorted = sorted(blue_probs.items(), key=lambda x: x[1], reverse=True)
        max_bp = blue_sorted[0][1] if blue_sorted else 1.0

        blue_grid = GridLayout(cols=8, spacing=dp(3),
                               size_hint_y=None, height=dp(36) * 2, padding=dp(2))
        for ball, prob in blue_sorted:
            blue_grid.add_widget(self._heat_cell(ball, prob, max_bp, False))
        self.result_box.add_widget(blue_grid)

    def _rec_row(self, rank, rec):
        row = FloatLayout(size_hint_y=None, height=dp(52))
        with row.canvas.before:
            bg_c = C_GOLD if rank == 1 else C_SURFACE2 if rank <= 3 else C_SURFACE
            Color(*bg_c)
            RoundedRectangle(pos=row.pos, size=row.size, radius=[dp(12)])
            if rank <= 3:
                Color(*C_BORDER)
                Line(rounded_rectangle=(*row.pos, *row.size, dp(12)), width=dp(0.7))
        row.bind(pos=lambda i,v: self._rc2(row), size=lambda i,v: self._rc2(row))

        # 排名徽章
        badge_c = C_GOLD if rank == 1 else C_PRIMARY if rank == 2 else C_ACCENT
        badge = Label(
            text=str(rank), font_size=sp(16), bold=True,
            color=(0,0,0,1) if rank == 1 else C_WHITE,
            pos_hint={'x': 0.02, 'center_y': 0.5},
            size_hint=(None, None), size=(dp(30), dp(30)),
            halign='center', valign='middle',
        )
        with badge.canvas.before:
            Color(*badge_c)
            Ellipse(pos=(badge.x + dp(1), badge.y + dp(1)),
                    size=(dp(28), dp(28)))
        badge.bind(pos=lambda i,v: _b(i, badge_c))
        def _b(w, c):
            for child in w.canvas.before.children:
                if isinstance(child, Ellipse):
                    child.pos = (w.x + dp(2), w.y + dp(2))
        row.add_widget(badge)

        # 红球
        for i, r in enumerate(rec["红球"]):
            bw = BallWidget(number=r, is_blue=False, ball_size=dp(32),
                            pos_hint={'x': 0.09 + i * 0.085, 'center_y': 0.5})
            row.add_widget(bw)

        # 蓝球
        bw_b = BallWidget(number=rec["蓝球"], is_blue=True, ball_size=dp(34),
                          pos_hint={'x': 0.62, 'center_y': 0.5})
        row.add_widget(bw_b)

        # 评分
        score_lbl = Label(
            text=f"{rec['combined_score']:.1f}",
            font_size=sp(11), color=C_TEXT2,
            pos_hint={'right': 0.96, 'center_y': 0.5},
            size_hint=(None, None), size=(dp(50), dp(20)),
            halign='right',
        )
        row.add_widget(score_lbl)

        return row

    def _heat_cell(self, ball, prob, max_prob, is_red):
        """权重热力格"""
        t = prob / max(max_prob, 0.001)
        if is_red:
            # 红: 浅灰 → 橙 → 深红
            lo = get_color_from_hex("#3A3A3A")
            hi = get_color_from_hex("#FF2222")
        else:
            # 蓝: 浅灰 → 青 → 深蓝
            lo = get_color_from_hex("#3A3A3A")
            hi = get_color_from_hex("#2266FF")
        bg = _lerp_color(lo, hi, t ** 0.6)
        fs = sp(10 + 5 * t)

        cell = Label(
            text=f"{str(ball).zfill(2)}\n{prob*100:.1f}%",
            font_size=fs, bold=t > 0.6, color=C_WHITE,
            size_hint=(None, None), width=dp(52), height=dp(34),
            halign='center', valign='middle',
        )
        with cell.canvas.before:
            Color(*bg)
            RoundedRectangle(pos=cell.pos, size=cell.size, radius=[dp(8)])
        cell.bind(pos=lambda i,v: _h(i),
                  size=lambda i,v: _h(i))
        def _h(w):
            w.canvas.before.clear()
            with w.canvas.before:
                Color(*bg)
                RoundedRectangle(pos=w.pos, size=w.size, radius=[dp(8)])
        return cell

    def _r(self, *a): self._bg.size = self.size; self._bg.pos = self.pos
    def _rc2(self, row):
        for c in row.canvas.before.children:
            if isinstance(c, (RoundedRectangle, Rectangle)):
                c.size = row.size; c.pos = row.pos


# ====================================================================
# App
# ====================================================================

class SSQApp(App):
    title = "SSQ Predictor"
    icon = "icon.png"

    def build(self):
        Window.clearcolor = C_BG
        sm = ScreenManager(transition=SlideTransition(duration=0.25))
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(HistoryScreen(name='history'))
        sm.add_widget(PredictScreen(name='predict'))
        return sm


if __name__ == '__main__':
    try:
        SSQApp().run()
    except Exception as e:
        print(f"Kivy error: {e}")
        sys.exit(1)
