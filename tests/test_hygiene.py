"""确定性简历体检（hygiene）检查器的测试。"""

from __future__ import annotations

from multi_agent_resume_screener.pipeline.hygiene import check_hygiene
from multi_agent_resume_screener.state import (
    EducationItem,
    ExperienceItem,
    ProjectItem,
    StructuredResume,
)


def _strong_resume() -> StructuredResume:
    return StructuredResume(
        name="Jane Smith",
        email="jane@example-mail.org",
        skills=["Python", "Go", "Docker", "Kubernetes", "PostgreSQL"],
        experience=[
            ExperienceItem(
                company="Acme",
                role="SWE Intern",
                bullets=["Reduced latency by 40% serving 10k requests/day"],
            )
        ],
        projects=[ProjectItem(name="RAG Knowledge Assistant", tech=["Python"])],
        education=[EducationItem(degree="B.Tech CS", institute="IIT")],
    )


def _strong_raw_text() -> str:
    return (
        "Jane Smith\njane@example-mail.org\n"
        "https://github.com/jane  https://linkedin.com/in/jane\n"
    )


def test_strong_resume_scores_high_with_no_warnings():
    report = check_hygiene(_strong_resume(), _strong_raw_text())
    warnings = [i for i in report.issues if i.severity == "warning"]
    assert warnings == []
    assert report.score >= 0.9
    assert report.positives  # 收集到了一些加分项


def test_missing_links_flagged():
    report = check_hygiene(_strong_resume(), raw_text="no links here")
    checks = {i.check for i in report.issues}
    assert "missing_github" in checks
    assert "no_links" in checks


def test_thin_skills_flagged():
    resume = _strong_resume()
    resume.skills = ["Python"]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "thin_skills" for i in report.issues)


def test_no_projects_flagged_as_warning():
    resume = _strong_resume()
    resume.projects = []
    report = check_hygiene(resume, _strong_raw_text())
    assert any(
        i.check == "no_projects" and i.severity == "warning"
        for i in report.issues
    )


def test_generic_project_name_flagged():
    resume = _strong_resume()
    resume.projects = [ProjectItem(name="Project 1")]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "generic_project_name" for i in report.issues)


def test_unquantified_experience_flagged():
    resume = _strong_resume()
    resume.experience = [
        ExperienceItem(company="Acme", role="Intern",
                       bullets=["Worked on backend services"])
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "unquantified_experience" for i in report.issues)


def test_placeholder_url_flagged():
    resume = _strong_resume()
    report = check_hygiene(resume, raw_text="see https://github.com/x and example.com")
    assert any(i.check == "placeholder_url" for i in report.issues)


def test_bare_domain_links_not_flagged_as_no_links():
    # 没有 http(s):// 前缀的链接（如 "github.com/x"）也算链接；
    # 当存在 GitHub/LinkedIn 链接时，不应触发 "no_links"。
    resume = _strong_resume()
    report = check_hygiene(resume, raw_text="github.com/jane linkedin.com/in/jane")
    checks = {i.check for i in report.issues}
    assert "no_links" not in checks
    assert "missing_github" not in checks


def test_score_never_negative():
    # 几乎为空的简历会累计许多扣分，但分数仍保持 >= 0。
    report = check_hygiene(StructuredResume(), raw_text="")
    assert 0.0 <= report.score <= 1.0


def test_deterministic():
    resume = _strong_resume()
    raw = _strong_raw_text()
    assert check_hygiene(resume, raw).score == check_hygiene(resume, raw).score


def test_weak_action_verb_flagged():
    resume = _strong_resume()
    resume.experience = [
        ExperienceItem(
            company="Acme",
            role="Intern",
            bullets=["Responsible for maintaining the deployment pipeline in 2023"],
        )
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "weak_action_verb" for i in report.issues)


def test_strong_verb_not_flagged_as_weak():
    # 优秀简历的条目以 "Reduced" 开头，不应被标记。
    report = check_hygiene(_strong_resume(), _strong_raw_text())
    assert not any(i.check == "weak_action_verb" for i in report.issues)


def test_long_bullet_flagged():
    resume = _strong_resume()
    long_bullet = "Built " + " ".join(f"thing{i}" for i in range(40)) + " achieving 10%"
    resume.experience = [
        ExperienceItem(company="Acme", role="Intern", bullets=[long_bullet])
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "long_bullet" for i in report.issues)


def test_first_person_flagged():
    resume = _strong_resume()
    resume.experience = [
        ExperienceItem(
            company="Acme",
            role="Intern",
            bullets=["Built my own service that cut costs by 20%"],
        )
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "first_person" for i in report.issues)


def test_first_person_ignores_ie_abbreviation():
    # "i.e." 不能被误认为是代词 "I"。
    resume = _strong_resume()
    resume.experience = [
        ExperienceItem(
            company="Acme",
            role="Intern",
            bullets=["Reduced latency, i.e. faster responses, by 40% across 3 apps"],
        )
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert not any(i.check == "first_person" for i in report.issues)


def test_buzzwords_flagged():
    resume = _strong_resume()
    resume.experience = [
        ExperienceItem(
            company="Acme",
            role="Intern",
            bullets=["A hardworking team player who reduced latency by 40%"],
        )
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "buzzwords" for i in report.issues)


def test_strong_resume_triggers_no_quickfix_rules():
    # 干净的简历不应触发任何招聘者快速修正规则。
    report = check_hygiene(_strong_resume(), _strong_raw_text())
    quickfix = {"weak_action_verb", "long_bullet", "first_person", "buzzwords"}
    assert not any(i.check in quickfix for i in report.issues)
