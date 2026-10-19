#!/usr/bin/env python3
"""Build the targeted Chinese resume as a compact two-page Word document.

The existing resume package is used only as a valid OOXML container and for
its embedded ID photo. The document body is rebuilt with explicit A4 geometry
and direct formatting so LibreOffice and Microsoft Word render it consistently.
"""

from __future__ import annotations

import copy
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def qn(prefix: str, name: str) -> str:
    return f"{{{NS[prefix]}}}{name}"


def add(parent: ET.Element, prefix: str, name: str, **attrs: str) -> ET.Element:
    element = ET.SubElement(parent, qn(prefix, name))
    for key, value in attrs.items():
        element.set(qn("w", key) if ":" not in key else qn(*key.split(":", 1)), str(value))
    return element


def run(
    text: str,
    *,
    size: int = 18,
    bold: bool = False,
    color: str = "172033",
    east_asia: str = "Microsoft YaHei",
    ascii_font: str = "Arial",
) -> ET.Element:
    element = ET.Element(qn("w", "r"))
    props = add(element, "w", "rPr")
    fonts = add(props, "w", "rFonts")
    fonts.set(qn("w", "ascii"), ascii_font)
    fonts.set(qn("w", "hAnsi"), ascii_font)
    fonts.set(qn("w", "eastAsia"), east_asia)
    add(props, "w", "sz", val=str(size))
    add(props, "w", "szCs", val=str(size))
    add(props, "w", "color", val=color)
    if bold:
        add(props, "w", "b")
        add(props, "w", "bCs")
    text_element = add(element, "w", "t")
    text_element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_element.text = text
    return element


def paragraph(
    parts: list[ET.Element] | None = None,
    *,
    before: int = 0,
    after: int = 0,
    line: int = 260,
    align: str | None = None,
    keep_next: bool = False,
    keep_lines: bool = True,
    left: int = 0,
    hanging: int = 0,
    bottom_border: bool = False,
) -> ET.Element:
    element = ET.Element(qn("w", "p"))
    props = add(element, "w", "pPr")
    spacing = add(props, "w", "spacing")
    spacing.set(qn("w", "before"), str(before))
    spacing.set(qn("w", "after"), str(after))
    spacing.set(qn("w", "line"), str(line))
    spacing.set(qn("w", "lineRule"), "auto")
    if align:
        add(props, "w", "jc", val=align)
    if keep_next:
        add(props, "w", "keepNext")
    if keep_lines:
        add(props, "w", "keepLines")
    if left or hanging:
        indentation = add(props, "w", "ind")
        indentation.set(qn("w", "left"), str(left))
        indentation.set(qn("w", "hanging"), str(hanging))
    if bottom_border:
        borders = add(props, "w", "pBdr")
        border = add(borders, "w", "bottom")
        border.set(qn("w", "val"), "single")
        border.set(qn("w", "sz"), "10")
        border.set(qn("w", "space"), "3")
        border.set(qn("w", "color"), "0B63A5")
    for item in parts or []:
        element.append(item)
    return element


def section_title(text: str, *, continued: bool = False) -> ET.Element:
    suffix = "（续）" if continued else ""
    return paragraph(
        [run(text + suffix, size=23, bold=True, color="0B4F78")],
        before=95,
        after=55,
        line=280,
        keep_next=True,
        bottom_border=True,
    )


def bullet(text: str) -> ET.Element:
    return paragraph(
        [run("•", size=17, bold=True, color="0B63A5"), run("  " + text, size=17)],
        after=28,
        line=238,
        left=300,
        hanging=210,
    )


def cell(width: int, children: list[ET.Element], *, valign: str = "center") -> ET.Element:
    element = ET.Element(qn("w", "tc"))
    props = add(element, "w", "tcPr")
    add(props, "w", "tcW", w=str(width), type="dxa")
    add(props, "w", "vAlign", val=valign)
    margins = add(props, "w", "tcMar")
    for side in ("top", "left", "bottom", "right"):
        node = add(margins, "w", side)
        node.set(qn("w", "w"), "0")
        node.set(qn("w", "type"), "dxa")
    for child in children:
        element.append(child)
    if not children:
        element.append(paragraph())
    return element


def table(rows: list[list[ET.Element]], widths: list[int], *, after: int = 0) -> ET.Element:
    element = ET.Element(qn("w", "tbl"))
    props = add(element, "w", "tblPr")
    add(props, "w", "tblW", w=str(sum(widths)), type="dxa")
    add(props, "w", "tblLayout", type="fixed")
    borders = add(props, "w", "tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = add(borders, "w", side)
        edge.set(qn("w", "val"), "nil")
    grid = add(element, "w", "tblGrid")
    for width in widths:
        add(grid, "w", "gridCol", w=str(width))
    for row_cells in rows:
        row = add(element, "w", "tr")
        for row_cell in row_cells:
            row.append(row_cell)
    if after:
        element.tail = ""
    return element


def photo_run() -> ET.Element:
    """Return a run containing the ID photo (ElementTree has no getparent)."""
    cx, cy = 828000, 1116000
    element = ET.Element(qn("w", "r"))
    drawing = add(element, "w", "drawing")
    inline = add(drawing, "wp", "inline")
    for key in ("distT", "distB", "distL", "distR"):
        inline.set(key, "0")
    extent = add(inline, "wp", "extent")
    extent.set("cx", str(cx)); extent.set("cy", str(cy))
    effect = add(inline, "wp", "effectExtent")
    for key in ("l", "t", "r", "b"):
        effect.set(key, "0")
    doc_pr = add(inline, "wp", "docPr")
    doc_pr.set("id", "1"); doc_pr.set("name", "ID Photo")
    frame = add(inline, "wp", "cNvGraphicFramePr")
    locks = add(frame, "a", "graphicFrameLocks"); locks.set("noChangeAspect", "1")
    graphic = add(inline, "a", "graphic")
    data = add(graphic, "a", "graphicData")
    data.set("uri", "http://schemas.openxmlformats.org/drawingml/2006/picture")
    picture = add(data, "pic", "pic")
    nv = add(picture, "pic", "nvPicPr")
    cnv = add(nv, "pic", "cNvPr"); cnv.set("id", "0"); cnv.set("name", "ID Photo")
    add(nv, "pic", "cNvPicPr")
    fill = add(picture, "pic", "blipFill")
    blip = add(fill, "a", "blip"); blip.set(qn("r", "embed"), "rId5")
    stretch = add(fill, "a", "stretch"); add(stretch, "a", "fillRect")
    shape = add(picture, "pic", "spPr")
    transform = add(shape, "a", "xfrm")
    offset = add(transform, "a", "off"); offset.set("x", "0"); offset.set("y", "0")
    ext = add(transform, "a", "ext"); ext.set("cx", str(cx)); ext.set("cy", str(cy))
    geometry = add(shape, "a", "prstGeom"); geometry.set("prst", "rect")
    add(geometry, "a", "avLst")
    return element


def project_header(title: str, role: str, dates: str) -> ET.Element:
    left = paragraph(
        [run(title, size=19, bold=True, color="102A43"), run("　" + role, size=17, bold=True, color="0B63A5")],
        after=22,
        line=235,
        keep_next=True,
    )
    right = paragraph(
        [run(dates, size=17, bold=True, color="486581")],
        after=22,
        line=235,
        align="right",
        keep_next=True,
    )
    return table([[cell(8650, [left]), cell(1750, [right])]], [8650, 1750])


def page_break() -> ET.Element:
    p = paragraph(keep_lines=False)
    r = add(p, "w", "r")
    br = add(r, "w", "br")
    br.set(qn("w", "type"), "page")
    return p


def build_document() -> bytes:
    root = ET.Element(qn("w", "document"))
    body = add(root, "w", "body")

    name_block = [
        paragraph([run("张大为", size=42, bold=True, color="102A43")], after=5, line=430, keep_next=True),
        paragraph([run("求职方向：机器人学习 / 模仿学习 / 强化学习", size=21, bold=True, color="0B63A5")], after=32, line=250, keep_next=True),
        paragraph([run("17635250138　|　davi1390723349@outlook.com", size=18, color="334E68")], line=220),
    ]
    photo_block = [paragraph([photo_run()], align="right", keep_lines=False)]
    body.append(table([[cell(8450, name_block), cell(1950, photo_block)]], [8450, 1950]))

    body.append(section_title("教育背景"))
    edu_rows = [
        [
            cell(2600, [paragraph([run("浙江大学", bold=True, size=18)], after=20, line=230)]),
            cell(5150, [paragraph([run("控制科学与工程　硕士研究生（在读）", size=18)], after=20, line=230)]),
            cell(2650, [paragraph([run("机器人学习 / 具身智能", size=17, color="486581")], after=20, line=230, align="right")]),
        ],
        [
            cell(2600, [paragraph([run("武汉大学", bold=True, size=18)], line=230)]),
            cell(5150, [paragraph([run("电子信息工程　本科", size=18)], line=230)]),
            cell(2650, [paragraph([run("GPA 3.73 / 4.0", size=17, color="486581")], line=230, align="right")]),
        ],
    ]
    body.append(table(edu_rows, [2600, 5150, 2650]))

    body.append(section_title("个人概述"))
    body.append(paragraph([run("具备机器人示教数据采集、模仿学习策略训练、GPU 并行强化学习与时序视觉控制的完整项目经验；熟悉从多传感器数据同步、数据集构建，到 ACT / Diffusion Policy / DAgger 策略训练、仿真评测和控制接口部署的流程。具备 UR5、Kinova Gen3 遥操作与策略实机部署经验，以及足式机器人全身控制、接触建模和视觉策略开发经验。", size=17)], after=25, line=235))

    body.append(section_title("专业技能"))
    skills = [
        ("编程与工程", "Python、PyTorch、NumPy，Linux，Git；多进程、共享内存、GPU 批量训练、实验配置与日志管理；C/C++ 基础。"),
        ("模仿学习", "熟悉 BC、ACT（action chunking / temporal ensembling）、Diffusion Policy、VQ-BeT、RDT；了解 flow matching、π0 / π0.5、DAgger 及人工介入式数据聚合。"),
        ("强化学习", "熟悉 PPO、非对称 Actor-Critic、多 Critic、课程学习、域随机化与策略蒸馏；了解 IQL、CQL、AWR 及离线数据分布偏移。"),
        ("机器人平台", "刚体动力学、运动学、坐标变换、轨迹规划、接触建模；ROS、TF、URDF、MoveIt；Isaac Lab/Gym、Legged Gym、MuJoCo、Gazebo。"),
        ("数据与评测", "多视角 RGB-D、关节状态、末端位姿和指令的时间戳采集与回放；时序窗口、接触/力信号特征、消融与对照实验。"),
    ]
    skill_rows = []
    for label, value in skills:
        skill_rows.append([
            cell(1550, [paragraph([run(label, size=17, bold=True, color="0B4F78")], after=16, line=225)]),
            cell(8850, [paragraph([run(value, size=16)], after=16, line=225)]),
        ])
    body.append(table(skill_rows, [1550, 8850]))

    body.append(section_title("项目经历"))
    body.append(project_header("华为校企合作｜具身智能机器人数据采集与策略学习", "项目负责人", "2024.09–2025.06"))
    for text in [
        "面向机械臂与双臂操作任务，负责示教采集、数据治理、策略训练和仿真/实机验证；使用 3D 鼠标与 Vision Pro 搭建遥操作链路，完成 UR5、Kinova Gen3 及灵巧手控制，并围绕 Push-T、双臂协作倒水等任务构建示教数据。",
        "设计多进程 + 共享内存数据采集框架，统一记录多视角 RGB-D、关节状态、末端位姿与控制指令；通过统一时钟、时间戳对齐、episode 元数据和离线回放检查，形成可复现的数据集构建与训练流水线。",
        "复现并对比 ACT 与 Diffusion Policy：ACT 采用 CVAE 生成 action chunk，并以 temporal ensembling 融合重叠预测；Diffusion Policy 采用视觉条件动作序列去噪与 receding-horizon 执行。围绕 chunk 长度、预测/执行窗口、观测历史、损失权重和推理延迟开展消融。",
        "研究 flow matching、π0 与 π0.5：理解“预训练 VLM 语义表征 + action expert 低层动作块”及“语义子任务预测 → flow action chunk”分层推理；结合液压机械臂控制频率、压力/力反馈、动作平滑性和人工接管需求，形成策略选型与数据闭环方案。",
    ]:
        body.append(bullet(text))

    body.append(project_header("LadderMan｜人形机器人视觉爬梯策略学习", "核心开发", "2026.06–至今"))
    for text in [
        "基于 Isaac Lab + RSL-RL 搭建 CR1 爬梯任务；完成 BVH 数据清洗、Holosoma 重定向、50 Hz 参考轨迹与手脚接触/梯级标签生成，并通过碰撞审计和有效帧掩码保证随机起始状态安全。",
        "在 4096 个 GPU 并行环境中训练状态专家，将全身动作跟踪与手脚目标接触、错误梯级碰撞、基座姿态等约束联合建模；同步随机化梯子倾角/横档间距及质量、质心、关节偏置和摩擦参数，完成多几何条件仿真闭环验证。",
        "构建视觉学生策略：以 60×80、30 Hz 头部深度图 + 本体历史为输入，采用 CNN/GRU 时序融合；使用冻结的全状态专家生成标签并实现 scheduled DAgger，使控制权从教师逐步过渡到学生，缓解闭环分布偏移。",
        "建立序列保持 rollout storage、按环境 GRU 状态重置、深度异步更新及训练/验证隔离；对纯 BC、混合 DAgger、视觉 PPO 与辅助感知头开展对照，记录动作误差、接触状态、爬升进度和失败模式。",
    ]:
        body.append(bullet(text))

    body.append(page_break())
    body.append(section_title("项目经历", continued=True))
    body.append(project_header("四足/轮足机器人足臂协同末端 6D 位姿控制", "核心开发", "2025.09–2026.05"))
    for text in [
        "基于 Isaac Gym / Legged Gym 搭建 M20-K8 移动操作环境，配置 6144 个 GPU 并行环境、22 维全身动作与本体历史观测；以位置 + 四元数命令及末端立方体关键点误差刻画 6D 位姿跟踪。",
        "针对“底盘接近目标”与“机械臂精确操作”的冲突，设计 RFM 风格连续阶段变量与 locomotion / manipulation 奖励门控；将 locomotion、manipulation、basic safety、stand 拆分为四个价值头，开发 Multi-Critic PPO，降低奖励间干扰。",
        "离线采样 1 万个机械臂工作空间点，按径向距离与关节难度分箱，构建从近距离易目标到远距离全姿态目标的课程；结合成功保持、超时惩罚和命令序列完成率，使策略在目标超出机械臂舒适区时主动调整底盘位置与朝向。",
        "加入摩擦、基座质量/质心、电机强度与 PD 增益随机化，在平地/粗糙地形评测末端误差、底盘漂移、轮滑、能耗、关节限位和命令成功率；完成固定机械臂、单任务与联合训练等对照实验和仿真闭环验证。",
    ]:
        body.append(bullet(text))

    body.append(project_header("全国大学生电子设计竞赛（英特尔杯）｜咽拭子采样机器人", "队长", "2022.03–2022.08"))
    for text in [
        "基于 ROS 完成室内服务机器人系统集成：使用 Cartographer 建图与定位、初始化位姿接口快速重定位，结合 Move_base 与 TEB 实现全局/局部无碰撞导航。",
        "基于 MoveIt 完成机械臂手眼标定、运动学求解与路径规划，结合视觉目标位姿实现视觉伺服、咽拭子抓取与口腔采样；负责系统联调、现场演示和答辩，获全国一等奖及应用创新奖。",
        "作为队长参加 2022 中国国际服务贸易交易会英特尔中国教育展台，负责机器人连续演示、技术讲解与异常处置。",
    ]:
        body.append(bullet(text))

    body.append(section_title("奖项与证书"))
    body.append(paragraph([run("全国大学生电子设计竞赛（英特尔杯嵌入式邀请赛）", size=17, bold=True), run("　全国一等奖、应用创新奖", size=17)], after=22, line=230))
    body.append(paragraph([run("英语能力", size=17, bold=True), run("　CET-6", size=17)], line=230))

    sect = add(body, "w", "sectPr")
    page_size = add(sect, "w", "pgSz")
    page_size.set(qn("w", "w"), "11906")
    page_size.set(qn("w", "h"), "16838")
    margins = add(sect, "w", "pgMar")
    margins.set(qn("w", "top"), "520")
    margins.set(qn("w", "right"), "680")
    margins.set(qn("w", "bottom"), "520")
    margins.set(qn("w", "left"), "680")
    margins.set(qn("w", "header"), "300")
    margins.set(qn("w", "footer"), "300")
    margins.set(qn("w", "gutter"), "0")
    add(sect, "w", "cols", space="425")
    add(sect, "w", "docGrid", linePitch="312")

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def core_properties() -> bytes:
    return b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>\xe5\xbc\xa0\xe5\xa4\xa7\xe4\xb8\xba\xef\xbd\x9c\xe6\x9c\xba\xe5\x99\xa8\xe4\xba\xba\xe5\xad\xa6\xe4\xb9\xa0\xe7\xa0\x94\xe7\xa9\xb6\xe5\xae\x9e\xe4\xb9\xa0\xe7\x94\x9f\xe5\xae\x9a\xe5\x90\x91\xe7\xae\x80\xe5\x8e\x86</dc:title><dc:creator>\xe5\xbc\xa0\xe5\xa4\xa7\xe4\xb8\xba</dc:creator><cp:lastModifiedBy>\xe5\xbc\xa0\xe5\xa4\xa7\xe4\xb8\xba</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">2026-09-23T10:00:00Z</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">2026-09-23T10:00:00Z</dcterms:modified><cp:revision>1</cp:revision></cp:coreProperties>'''


def app_properties() -> bytes:
    return b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Template>Normal</Template><Pages>2</Pages><Application>Microsoft Office Word</Application><DocSecurity>0</DocSecurity></Properties>'''


def custom_properties() -> bytes:
    return b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"/>'''


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_resume_docx.py TEMPLATE.docx OUTPUT.docx")
    template = Path(sys.argv[1])
    output = Path(sys.argv[2])
    document_xml = build_document()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(template, "r") as source, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            if item.filename == "word/document.xml":
                target.writestr(item, document_xml)
            elif item.filename == "docProps/core.xml":
                target.writestr(item, core_properties())
            elif item.filename == "docProps/app.xml":
                target.writestr(item, app_properties())
            elif item.filename == "docProps/custom.xml":
                target.writestr(item, custom_properties())
            else:
                target.writestr(item, source.read(item.filename))


if __name__ == "__main__":
    main()
