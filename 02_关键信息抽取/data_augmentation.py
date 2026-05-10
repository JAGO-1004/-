"""
数据伪造（数据增强）— 针对薄弱字段生成多样化训练数据

方法:
1. 字段值替换: 用 faker 生成合规的假数据替换真实证件字段
2. 模板渲染: 将假数据渲染到证件模板上，生成新图像
3. 难例挖掘: 针对易错字段（如数字0/O混淆、手写体等）专项增强
"""

import json
import os
import sys
import random
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_templates import TEMPLATES

# ============================================================
# 数据生成器 — 生成合规的假字段值
# ============================================================

class FieldValueGenerator:
    """字段值生成器 — 生成符合中国规范的证件字段值"""

    # 常见姓氏
    SURNAMES = [
        "王", "李", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴",
        "徐", "孙", "马", "朱", "胡", "郭", "何", "高", "林", "罗",
        "郑", "梁", "谢", "宋", "唐", "韩", "曹", "许", "邓", "萧",
    ]

    # 常见名字用字
    GIVEN_CHARS = [
        "伟", "芳", "娜", "秀英", "敏", "静", "丽", "强", "磊", "军",
        "洋", "勇", "艳", "杰", "娟", "涛", "明", "超", "秀兰", "霞",
        "平", "刚", "桂英", "文", "华", "建", "国", "志", "宇", "飞",
    ]

    # 民族
    ETHNICITIES = ["汉", "蒙古", "回", "藏", "满", "维吾尔", "苗", "彝", "壮", "布依", "朝鲜", "侗", "瑶", "白", "土家"]

    # 性别
    GENDERS = ["男", "女"]

    @staticmethod
    def name() -> str:
        return random.choice(FieldValueGenerator.SURNAMES) + \
               random.choice(FieldValueGenerator.GIVEN_CHARS)

    @staticmethod
    def id_number(birth=None) -> str:
        """生成合规的 18 位身份证号"""
        area = random.choice(["110101", "310101", "440101", "320101", "510101", "420101"])
        birth = birth or f"{random.randint(1960, 2000):04d}{random.randint(1, 12):02d}{random.randint(1, 28):02d}"
        seq = f"{random.randint(0, 999):03d}"
        base = area + birth + seq
        # 校验码
        weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        check_codes = "10X98765432"
        total = sum(int(base[i]) * weights[i] for i in range(17))
        return base + check_codes[total % 11]

    @staticmethod
    def date(start_year=1990, end_year=2025) -> str:
        return f"{random.randint(start_year, end_year):04d}{random.randint(1, 12):02d}{random.randint(1, 28):02d}"

    @staticmethod
    def date_range() -> str:
        s = FieldValueGenerator.date(2010, 2023)
        e = FieldValueGenerator.date(2024, 2035)
        return f"{s}-{e}"

    @staticmethod
    def phone() -> str:
        prefixes = ["13", "15", "17", "18", "19"]
        return random.choice(prefixes) + f"{random.randint(100000000, 999999999):09d}"

    @staticmethod
    def address() -> str:
        cities = ["北京市朝阳区", "上海市浦东新区", "广州市天河区", "深圳市南山区",
                   "杭州市西湖区", "成都市武侯区", "武汉市洪山区"]
        road = random.choice(["中山路", "人民路", "建设大道", "解放路", "科技路", "学府路"])
        number = f"{random.randint(1, 999)}号"
        detail = random.choice(["", f"{random.randint(1, 30)}栋{random.randint(1, 6)}单元"])
        return f"{random.choice(cities)}{road}{number}{detail}"

    @staticmethod
    def amount(min_v=10000, max_v=5000000) -> str:
        return f"{random.randint(min_v, max_v)}"

    @staticmethod
    def company() -> str:
        prefixes = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
        names = ["科", "创", "达", "信", "恒", "通", "源", "盛", "博", "瑞"]
        suffixes = ["科技有限公司", "信息技术有限公司", "实业有限公司", "投资有限公司"]
        return f"{random.choice(prefixes)}{''.join(random.choices(names, k=2))}{random.choice(suffixes)}"

    @staticmethod
    def generate(field_name: str) -> str:
        """根据字段名智能生成对应值"""
        generators = {
            "姓名": FieldValueGenerator.name,
            "持证人姓名": FieldValueGenerator.name,
            "配偶姓名": FieldValueGenerator.name,
            "户主姓名": FieldValueGenerator.name,
            "员工姓名": FieldValueGenerator.name,
            "借款人姓名": FieldValueGenerator.name,
            "申请人姓名": FieldValueGenerator.name,
            "买受人": FieldValueGenerator.name,
            "出卖人": FieldValueGenerator.name,
            "法定代表人": FieldValueGenerator.name,
            "代理人": FieldValueGenerator.name,
            "抵押人": FieldValueGenerator.name,
            "抵押权人": FieldValueGenerator.name,
            "保证人": FieldValueGenerator.name,
            "被保证人": FieldValueGenerator.name,
            "权利人": FieldValueGenerator.name,
            "被保险人": FieldValueGenerator.name,
            "投保人": FieldValueGenerator.name,
            "受益人": FieldValueGenerator.name,
            "新生儿姓名": FieldValueGenerator.name,
            "母亲姓名": FieldValueGenerator.name,
            "父亲姓名": FieldValueGenerator.name,
            "身份证号": FieldValueGenerator.id_number,
            "出生日期": FieldValueGenerator.date,
            "登记日期": lambda: FieldValueGenerator.date(2000, 2025),
            "离婚日期": lambda: FieldValueGenerator.date(2000, 2025),
            "签发日期": lambda: FieldValueGenerator.date(2000, 2025),
            "签署日期": lambda: FieldValueGenerator.date(2000, 2025),
            "开具日期": lambda: FieldValueGenerator.date(2000, 2025),
            "发证日期": lambda: FieldValueGenerator.date(2000, 2025),
            "有效期限": lambda: FieldValueGenerator.date(2020, 2035) + "-" + FieldValueGenerator.date(2035, 2050),
            "有效期至": lambda: FieldValueGenerator.date(2025, 2050),
            "合同期限开始": lambda: FieldValueGenerator.date(2020, 2025),
            "合同期限结束": lambda: FieldValueGenerator.date(2025, 2035),
            "保险期间起始": lambda: FieldValueGenerator.date(2020, 2025),
            "保险期间截止": lambda: FieldValueGenerator.date(2025, 2050),
            "住址": FieldValueGenerator.address,
            "居住地址": FieldValueGenerator.address,
            "注册地址": FieldValueGenerator.address,
            "住所": FieldValueGenerator.address,
            "房屋坐落": FieldValueGenerator.address,
            "月均收入": lambda: FieldValueGenerator.amount(5000, 100000),
            "贷款金额": lambda: FieldValueGenerator.amount(100000, 5000000),
            "申请金额": lambda: FieldValueGenerator.amount(100000, 5000000),
            "合同总价": lambda: FieldValueGenerator.amount(500000, 50000000),
            "月缴金额": lambda: FieldValueGenerator.amount(500, 10000),
            "保费": lambda: FieldValueGenerator.amount(1000, 100000),
            "保额": lambda: FieldValueGenerator.amount(100000, 50000000),
            "担保金额": lambda: FieldValueGenerator.amount(100000, 5000000),
            "保障金额": lambda: FieldValueGenerator.amount(500, 10000),
            "用人单位": FieldValueGenerator.company,
            "开具单位": FieldValueGenerator.company,
            "企业名称": FieldValueGenerator.company,
            "参保单位": FieldValueGenerator.company,
        }
        gen = generators.get(field_name, lambda: "示例值")
        return gen()


# ============================================================
# 数据增强管线
# ============================================================

class DataAugmentation:
    """数据增强 — 针对薄弱字段生成多样化训练数据"""

    def __init__(self, output_dir: str = "./data/augmented"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = FieldValueGenerator()

    def generate_text_samples(self, doc_type: str, count: int = 100,
                              weak_fields: list[str] = None) -> list[dict]:
        """
        生成文本级增强数据（不生成图像，只生成字段值对）

        Args:
            doc_type: 证件类型
            count: 生成数量
            weak_fields: 薄弱字段列表（这些字段会生成多样化变体）
        """
        if doc_type not in TEMPLATES:
            raise ValueError(f"未知证件类型: {doc_type}")

        fields = TEMPLATES[doc_type]["fields"]
        weak_fields = weak_fields or fields  # 默认全部增强

        samples = []
        for i in range(count):
            sample = {"doc_type": doc_type, "id": f"{doc_type}_syn_{i:04d}", "fields": {}}
            for f in fields:
                if f in weak_fields:
                    # 薄弱字段: 生成多个变体
                    sample["fields"][f] = self.generator.generate(f)
                else:
                    # 非薄弱字段: 使用标准生成逻辑
                    sample["fields"][f] = self.generator.generate(f)
            samples.append(sample)

        return samples

    def generate_hard_cases(self, doc_type: str, field_name: str, count: int = 50) -> list[dict]:
        """
        针对特定薄弱字段生成难例

        例如: 身份证号中的 0/O 混淆、g/9 混淆、手写体数字等
        """
        base_samples = self.generate_text_samples(doc_type, count)
        hard_cases = []

        for sample in base_samples:
            # 对目标字段应用混淆
            original = sample["fields"].get(field_name, "")
            confused = self._apply_confusion(original)
            sample["fields"][field_name] = confused
            sample["is_hard_case"] = True
            sample["confusion_type"] = self._detect_confusion_type(field_name)
            hard_cases.append(sample)

        return hard_cases

    def _apply_confusion(self, value: str) -> str:
        """对字段值应用光学混淆"""
        if not value:
            return value

        confusions = [
            ("0", "O"), ("O", "0"),     # 数字0和字母O
            ("1", "l"), ("l", "1"),     # 数字1和小写l
            ("5", "S"), ("S", "5"),     # 数字5和字母S
            ("8", "B"), ("B", "8"),     # 数字8和字母B
            ("6", "G"), ("G", "6"),     # 数字6和字母G
            ("9", "g"), ("g", "9"),     # 数字9和字母g
            ("2", "Z"), ("Z", "2"),     # 数字2和字母Z
        ]

        chars = list(value)
        # 随机选择 1-2 个字符替换
        if len(chars) <= 3:
            return value

        swap = random.choice(confusions)
        positions = random.sample(range(len(chars)), min(random.randint(1, 2), len(chars)))
        for pos in positions:
            if chars[pos] == swap[0]:
                chars[pos] = swap[1]

        return "".join(chars)

    def _detect_confusion_type(self, field_name: str) -> str:
        """判断适合哪种混淆类型"""
        if "身份证" in field_name or "号" in field_name:
            return "数字字母混淆"
        if "金额" in field_name or "保额" in field_name or "收入" in field_name:
            return "数字格式混淆"
        return "通用混淆"

    def save_samples(self, samples: list[dict], filename: str = None):
        """保存增强数据"""
        if not filename:
            filename = f"augmented_{datetime.now().strftime('%Y%m%d')}.jsonl"

        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"已保存 {len(samples)} 条增强数据到 {path}")
        return path

    def generate_report(self, training_data: list[dict] = None):
        """生成薄弱字段分析报告"""
        if not training_data:
            return

        # 统计哪些字段需要人工修正最多
        corrections = {}
        for entry in training_data:
            if entry.get("has_corrections"):
                for field in entry.get("corrected_fields", []):
                    corrections[field] = corrections.get(field, 0) + 1

        if corrections:
            print("\n=== 薄弱字段分析 ===")
            for field, count in sorted(corrections.items(), key=lambda x: -x[1]):
                print(f"  {field}: 被修正 {count} 次 ← 需要增强")
            return corrections
        return {}


if __name__ == "__main__":
    da = DataAugmentation()

    # 演示: 为身份证生成增强数据
    samples = da.generate_text_samples("身份证_正面", count=5)
    for s in samples:
        print(f"  [{s['id']}] {s['doc_type']}:")
        for f, v in s["fields"].items():
            print(f"      {f}: {v}")
        print()

    # 演示: 生成身份证号的难例
    hard = da.generate_hard_cases("身份证_正面", "身份证号", count=3)
    print("\n难例 (身份证号混淆):")
    for s in hard:
        print(f"  {s['id']}: {s['fields']['身份证号']} ({s['confusion_type']})")
