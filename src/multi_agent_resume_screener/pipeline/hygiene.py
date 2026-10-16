"""确定性的简历体检（hygiene）检查。

客观、零 LLM 的检查，用于标记与任何职位描述无关的常见简历问题：
缺少联系方式链接、经历未量化、项目名泛泛、技能列表单薄等等。结果是
建议性的——它为 critic 的建议提供参考并展示给候选人，但不会改变
JD 匹配分数。

由于是确定性的，这些检查成本低、可复现、易于辩护（"我们把客观的规则
检查与 LLM 判断相结合，而不是一个大提示词"）。这一概念受 HackerRank
开源招聘智能体加分/扣分思路的启发；这里的检查是独立、净室的实现。
"""

from __future__ import annotations

import re

from multi_agent_resume_screener.state import (
    HygieneIssue,
    HygieneReport,
    StructuredResume,
)

# 按严重程度对每个问题从体检分数中扣除的罚分。
_PENALTY = {"warning": 0.15, "info": 0.05}

# 技能数量低于该值即认为列表"单薄"。
_MIN_SKILLS = 5

_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
_GITHUB_RE = re.compile(r"github\.com/", re.IGNORECASE)
_LINKEDIN_RE = re.compile(r"linkedin\.com/", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")
_PLACEHOLDER_RE = re.compile(
    r"example\.com|localhost|your-?website|todo|lorem ipsum|xxx", re.IGNORECASE
)
_GENERIC_NAME_RE = re.compile(
    r"^\s*(project\s*\d*|final\s*(year)?\s*project|minor\s*project|"
    r"major\s*project|untitled|test|demo|sample)\s*$",
    re.IGNORECASE,
)

# 招聘者视角的"马上可修"规则（候选人现在就能在简历上改掉的问题）。
# 把影响力埋在被动/含糊措辞后面的无力开头。
_WEAK_OPENERS: tuple[str, ...] = (
    "worked on",
    "responsible for",
    "responsibilities included",
    "responsibilities include",
    "duties included",
    "helped with",
    "helped to",
    "assisted with",
    "assisted in",
    "involved in",
    "participated in",
    "tasked with",
    "in charge of",
    "was part of",
)
# 第一人称代词违反简历惯例（简历使用隐含的第一人称）。
# 区分大小写的 "I" 避免匹配到 "i.e." 中的 "i"；"US"（国家）被排除。
_FIRST_PERSON_RE = re.compile(
    r"\bI\b|\bI['\u2019](?:m|ve|ll|d)\b|\b(?:[Mm]y|[Mm]e|[Ww]e|[Oo]ur)\b"
)
# 招聘者会打折扣的空洞流行词；应替换为具体、量化的证据。
_BUZZWORDS: tuple[str, ...] = (
    "hardworking",
    "hard-working",
    "team player",
    "go-getter",
    "go getter",
    "detail-oriented",
    "detail oriented",
    "self-motivated",
    "self motivated",
    "results-driven",
    "results driven",
    "think outside the box",
    "fast learner",
    "quick learner",
    "passionate about",
    "synergy",
)
# 超过这个词数的条目在 6 秒快速浏览中很难扫读。
_LONG_BULLET_WORDS = 30


def check_hygiene(
    resume: StructuredResume,
    raw_text: str = "",
) -> HygieneReport:
    """对一份简历运行确定性的体检检查。

    Args:
        resume: 结构化简历。
        raw_text: 简历的原始文本（用于 URL/链接检测，在原始文本上比在
            结构化字段上更可靠）。

    Returns:
        一个 :class:`HygieneReport`，包含 0-1 的体检分数、客观问题列表
        和亮点列表。
    """
    issues: list[HygieneIssue] = []
    positives: list[str] = []

    # 合并文本让链接检查有最大机会找到 URL。
    combined = f"{raw_text}\n{resume.model_dump_json()}"

    _check_contact(resume, combined, issues, positives)
    _check_links(combined, issues, positives)
    _check_placeholder_urls(combined, issues)
    _check_skills(resume, issues, positives)
    _check_projects(resume, issues, positives)
    _check_quantified_experience(resume, issues, positives)
    _check_weak_action_verbs(resume, issues)
    _check_long_bullets(resume, issues)
    _check_first_person(resume, issues)
    _check_buzzwords(resume, issues)
    _check_education(resume, issues)

    penalty = sum(_PENALTY[i.severity] for i in issues)
    hygiene_score = max(0.0, min(1.0, 1.0 - penalty))

    return HygieneReport(score=hygiene_score, issues=issues, positives=positives)


def _add(issues: list[HygieneIssue], check: str, severity: str, message: str) -> None:
    issues.append(HygieneIssue(check=check, severity=severity, message=message))


def _check_contact(resume, combined, issues, positives) -> None:
    if not resume.email and "@" not in combined:
        _add(
            issues,
            "missing_email",
            "warning",
            "No email address found; recruiters need a way to contact you.",
        )
    elif resume.email:
        positives.append("Includes a contact email.")


def _check_links(combined, issues, positives) -> None:
    has_github = bool(_GITHUB_RE.search(combined))
    has_linkedin = bool(_LINKEDIN_RE.search(combined))
    has_any_url = bool(_URL_RE.search(combined))

    if has_github:
        positives.append("Links to a GitHub profile/projects.")
    if has_linkedin:
        positives.append("Links to a LinkedIn profile.")

    if not has_github:
        _add(
            issues,
            "missing_github",
            "warning",
            "No GitHub link found; add one so reviewers can see your code.",
        )
    if not has_linkedin:
        _add(
            issues,
            "missing_linkedin",
            "info",
            "No LinkedIn link found; consider adding your profile URL.",
        )
    if not has_any_url and not has_github and not has_linkedin:
        _add(
            issues,
            "no_links",
            "warning",
            "No links at all; add GitHub, a portfolio, or live project URLs.",
        )


def _check_placeholder_urls(combined, issues) -> None:
    if _PLACEHOLDER_RE.search(combined):
        _add(
            issues,
            "placeholder_url",
            "warning",
            "Found a placeholder/example link (e.g. example.com or TODO); "
            "replace it with a real URL.",
        )


def _check_skills(resume, issues, positives) -> None:
    n = len(resume.skills)
    if n == 0:
        _add(
            issues,
            "no_skills",
            "warning",
            "No skills listed; add a skills section with concrete technologies.",
        )
    elif n < _MIN_SKILLS:
        _add(
            issues,
            "thin_skills",
            "info",
            f"Only {n} skill(s) listed; consider expanding to show breadth.",
        )
    else:
        positives.append(f"Lists {n} skills.")


def _check_projects(resume, issues, positives) -> None:
    if not resume.projects:
        _add(
            issues,
            "no_projects",
            "warning",
            "No projects listed; projects are strong signal for most roles.",
        )
        return

    generic = [
        p.name
        for p in resume.projects
        if p.name and _GENERIC_NAME_RE.match(p.name)
    ]
    if generic:
        _add(
            issues,
            "generic_project_name",
            "info",
            "Generic project name(s) found "
            f"({', '.join(generic)}); use specific, descriptive titles.",
        )
    else:
        positives.append("Project names are specific.")


def _check_quantified_experience(resume, issues, positives) -> None:
    if not resume.experience:
        return  # 缺少经历由 matcher 判断，而非体检检查。

    has_quantified = any(
        _DIGIT_RE.search(bullet)
        for exp in resume.experience
        for bullet in exp.bullets
    )
    if has_quantified:
        positives.append("Experience bullets include quantified impact.")
    else:
        _add(
            issues,
            "unquantified_experience",
            "info",
            "Experience lacks numbers; quantify impact (e.g. 'cut latency 40%').",
        )


def _experience_bullets(resume) -> list[str]:
    """汇总所有经历条目下非空的 bullet。"""
    return [
        bullet
        for exp in resume.experience
        for bullet in exp.bullets
        if bullet and bullet.strip()
    ]


def _check_weak_action_verbs(resume, issues) -> None:
    offenders: list[str] = []
    for bullet in _experience_bullets(resume):
        opener = bullet.strip().lstrip("-*\u2022 \t").lower()
        if any(opener.startswith(weak) for weak in _WEAK_OPENERS):
            offenders.append(bullet.strip())
    if offenders:
        _add(
            issues,
            "weak_action_verb",
            "info",
            "Start bullets with a strong action verb (Built, Led, Reduced, "
            "Shipped) instead of weak openers like 'Worked on' or 'Responsible "
            f"for' ({len(offenders)} bullet(s)).",
        )


def _check_long_bullets(resume, issues) -> None:
    long_count = sum(
        1 for bullet in _experience_bullets(resume)
        if len(bullet.split()) > _LONG_BULLET_WORDS
    )
    if long_count:
        _add(
            issues,
            "long_bullet",
            "info",
            f"Condense {long_count} overly long bullet(s) to a single scannable "
            f"line (aim for under {_LONG_BULLET_WORDS} words).",
        )


def _check_first_person(resume, issues) -> None:
    texts = _experience_bullets(resume) + [
        p.description for p in resume.projects if p.description
    ]
    if any(_FIRST_PERSON_RE.search(t) for t in texts):
        _add(
            issues,
            "first_person",
            "info",
            "Remove first-person pronouns (I, my, we, our); resumes use the "
            "implied first person.",
        )


def _check_buzzwords(resume, issues) -> None:
    texts = _experience_bullets(resume) + [
        p.description for p in resume.projects if p.description
    ]
    haystack = " ".join(texts).lower()
    found = sorted({bw for bw in _BUZZWORDS if bw in haystack})
    if found:
        _add(
            issues,
            "buzzwords",
            "info",
            "Replace vague buzzwords ("
            f"{', '.join(found)}) with concrete, quantified achievements.",
        )


def _check_education(resume, issues) -> None:
    if not resume.education:
        _add(
            issues,
            "missing_education",
            "info",
            "No education section detected.",
        )
