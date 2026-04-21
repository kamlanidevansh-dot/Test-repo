export type JourneyStageData = {
  id: string;
  label: string;
  kicker: string;
  title: string;
  body: string;
  bullets: string[];
  stat?: { value: string; caption: string };
  dualStats?: { value: string; caption: string }[];
  tags: string[];
  quote?: string;
  attribution?: string;
};

export const hero = {
  eyebrow: "ENTERPRISE INNOVATION ACCELERATION",
  headline: "The gap between your innovation ambition and your results is a design problem.",
  highlight: "We fix it.",
  subtext:
    "We embed innovation capability, agentic AI, and decision intelligence into your enterprise. And we stay invested until the outcome is real.",
  primaryCta: "Book a Free Diagnostic",
  secondaryCta: "See the Journey",
};

export const stages: JourneyStageData[] = [
  {
    id: "01",
    label: "01",
    kicker: "Stage One",
    title: "Discover",
    body: "We dig deep. Not a 30-minute intake call. A proper diagnostic of your innovation readiness, your business gaps, and the exact levers that will move the needle fastest.",
    bullets: [
      "Innovation readiness audit across your organisation",
      "Stakeholder interviews and culture assessment",
      "Opportunity mapping against your biggest P&L drivers",
    ],
    stat: { value: "60min", caption: "Free diagnostic. No obligation." },
    tags: ["Innovation Audit", "Gap Analysis", "Opportunity Map"],
    quote: "The diagnostic revealed we were solving the wrong problems entirely.",
    attribution: "VP Operations, Ensono",
  },
  {
    id: "02",
    label: "02",
    kicker: "Stage Two",
    title: "Engage",
    body: "Strategy without ownership is wallpaper. We co-create the roadmap with your team so there is no adoption resistance later. This is your plan. We just built it with you.",
    bullets: [
      "Co-creation of your innovation capability roadmap",
      "VELOCITY™ program design for your specific context",
      "Executive alignment and change sponsorship",
    ],
    stat: { value: "100%", caption: "Co-created with your people, not handed down" },
    tags: ["VELOCITY™ Design", "Roadmap", "Exec Alignment"],
  },
  {
    id: "03",
    label: "03",
    kicker: "Stage Three",
    title: "Develop",
    body: "We build. Not PowerPoints. Real products: automation bots, innovation programs, analytics platforms. If it is on the roadmap, we ship it.",
    bullets: [
      "Lokibots: agentic automation for finance, HR and ops",
      "xLytix: predictive analytics and leadership dashboards",
      "Innovation programs built for your specific workforce",
    ],
    stat: { value: "250%+", caption: "TAT reduction via Lokibots automation" },
    tags: ["Lokibots", "xLytix", "VELOCITY™"],
  },
  {
    id: "04",
    label: "04",
    kicker: "Stage Four",
    title: "Adopt",
    body: "Most innovation dies in deployment. We drive the change management, embed the new behaviours, and make sure your people actually use what was built. Adoption is not optional.",
    bullets: [
      "Change management and behavioural embedding",
      "21% average reduction in attrition post-program",
      "200% average improvement in team engagement",
    ],
    dualStats: [
      { value: "200%", caption: "Engagement" },
      { value: "21%", caption: "Attrition drop" },
    ],
    tags: ["Change Mgmt", "Behaviour Design", "Culture Shift"],
  },
  {
    id: "05",
    label: "05",
    kicker: "Stage Five",
    title: "Scale Up",
    body: "One win is a proof of concept. We amplify it across your organisation, turning isolated success into enterprise-wide momentum. USD 30M in documented savings later, this is where it gets real.",
    bullets: [
      "Scale proven models across every business unit",
      "Build internal innovation champions who carry it forward",
      "Measure, report, and compound the results",
    ],
    stat: { value: "USD 30M+", caption: "Documented Savings" },
    tags: ["Scale", "Champions", "Compounding Results"],
  },
];
