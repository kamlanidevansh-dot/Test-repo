import { motion, useReducedMotion } from "framer-motion";
import type { JourneyStageData } from "@/lib/journeyData";

type JourneyStageProps = {
  stage: JourneyStageData;
  index: number;
  activeIndex: number;
};

export function JourneyStage({ stage, index, activeIndex }: JourneyStageProps) {
  const isActive = activeIndex === index;
  const reducedMotion = useReducedMotion();

  return (
    <motion.article
      className={`stage-card relative rounded-3xl border p-8 md:p-10 ${
        isActive ? "border-ember/70 bg-white/10" : "border-white/10 bg-white/[0.03]"
      }`}
      initial={{ opacity: 0.4, y: 20 }}
      animate={{
        opacity: isActive ? 1 : 0.55,
        y: 0,
        scale: isActive ? 1 : 0.985,
      }}
      transition={{ duration: reducedMotion ? 0 : 0.45, ease: "easeOut" }}
      aria-current={isActive ? "step" : undefined}
    >
      <div className="mb-5 flex items-center gap-3 text-xs uppercase tracking-[0.22em] text-white/65">
        <span className="rounded-full border border-white/20 px-3 py-1">{stage.label}</span>
        <span>{stage.kicker}</span>
      </div>
      <h3 className="text-3xl text-white md:text-4xl">{stage.title}</h3>
      <p className="mt-4 max-w-2xl text-base leading-relaxed text-white/78">{stage.body}</p>

      <ul className="mt-6 space-y-2 text-sm text-white/82">
        {stage.bullets.map((item) => (
          <li key={item} className="flex gap-3">
            <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-ember" />
            <span>{item}</span>
          </li>
        ))}
      </ul>

      {stage.stat ? (
        <div className="mt-8 rounded-2xl border border-white/10 bg-black/25 p-4">
          <p className="text-3xl font-semibold text-white">{stage.stat.value}</p>
          <p className="text-xs uppercase tracking-wider text-white/65">{stage.stat.caption}</p>
        </div>
      ) : null}

      {stage.dualStats ? (
        <div className="mt-8 grid gap-3 sm:grid-cols-2">
          {stage.dualStats.map((s) => (
            <div key={s.caption} className="rounded-2xl border border-white/10 bg-black/25 p-4">
              <p className="text-3xl font-semibold text-white">{s.value}</p>
              <p className="text-xs uppercase tracking-wider text-white/65">{s.caption}</p>
            </div>
          ))}
        </div>
      ) : null}

      <div className="mt-7 flex flex-wrap gap-2">
        {stage.tags.map((tag) => (
          <span key={tag} className="rounded-full border border-white/20 bg-white/5 px-3 py-1 text-xs text-white/70">
            {tag}
          </span>
        ))}
      </div>

      {stage.quote ? (
        <blockquote className="mt-7 border-l border-ember/70 pl-4 text-sm italic text-white/78">
          “{stage.quote}”
          {stage.attribution ? <footer className="mt-2 not-italic text-xs text-white/55">{stage.attribution}</footer> : null}
        </blockquote>
      ) : null}
    </motion.article>
  );
}
