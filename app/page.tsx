"use client";

import { useMemo, useRef, useState } from "react";
import {
  motion,
  useMotionValueEvent,
  useReducedMotion,
  useScroll,
  useTransform,
} from "framer-motion";
import { CTAButton } from "@/components/CTAButton";
import { JourneyScene } from "@/components/JourneyScene";
import { JourneyStage } from "@/components/JourneyStage";
import { hero, stages } from "@/lib/journeyData";

export default function HomePage() {
  const sectionRef = useRef<HTMLElement>(null);
  const reducedMotion = useReducedMotion();
  const [progressValue, setProgressValue] = useState(0);

  // Scroll is normalized across the entire journey section and mapped to visual progression.
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start start", "end end"],
  });
  const progress = useTransform(scrollYProgress, [0, 1], [0, 1]);

  useMotionValueEvent(progress, "change", (latest) => setProgressValue(latest));

  const stageFractions = useMemo(() => stages.map((_, i) => i / (stages.length - 1)), []);
  const activeIndex = Math.min(
    stages.length - 1,
    Math.max(0, Math.round(progressValue * (stages.length - 1)))
  );

  return (
    <main className="relative overflow-x-hidden">
      <section className="relative flex min-h-[92vh] items-center px-6 py-24 md:px-10 lg:px-16">
        <div className="mx-auto grid w-full max-w-7xl gap-12 lg:grid-cols-[1.1fr_0.9fr] lg:items-end">
          <div>
            <p className="text-xs tracking-[0.28em] text-white/65">{hero.eyebrow}</p>
            <h1 className="mt-6 max-w-4xl text-4xl leading-tight text-white md:text-6xl md:leading-[1.1]">
              {hero.headline}
            </h1>
            <p className="mt-4 text-3xl text-ember md:text-5xl">{hero.highlight}</p>
            <p className="mt-8 max-w-2xl text-lg leading-relaxed text-white/78">{hero.subtext}</p>
            <div className="mt-10 flex flex-wrap gap-4">
              <CTAButton href="#journey">{hero.primaryCta}</CTAButton>
              <CTAButton href="#journey" variant="secondary">
                {hero.secondaryCta}
              </CTAButton>
            </div>
          </div>
          <motion.div
            className="rounded-3xl border border-white/10 bg-white/[0.03] p-6"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: reducedMotion ? 0 : 0.8 }}
          >
            <p className="text-sm text-white/70">Journey progression</p>
            <div className="mt-5 flex gap-3">
              {stageFractions.map((_, i) => (
                <div key={stages[i].id} className="h-2 flex-1 rounded-full bg-white/10">
                  <motion.div
                    className="h-2 rounded-full bg-gradient-to-r from-[#ff4d2d] to-[#ff7b51]"
                    animate={{ width: activeIndex >= i ? "100%" : "0%" }}
                  />
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      <section id="journey" ref={sectionRef} className="relative mx-auto max-w-7xl px-6 pb-24 md:px-10 lg:px-16">
        <div className="grid gap-8 lg:grid-cols-[0.94fr_1.06fr]">
          <div className="lg:sticky lg:top-6 lg:h-[92vh]">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_40%,rgba(255,88,58,0.15),transparent_52%)]" />
            <JourneyScene
              progress={progressValue}
              activeIndex={activeIndex}
              reducedMotion={!!reducedMotion}
            />
          </div>

          <div className="space-y-12 pb-28 lg:pt-10">
            {stages.map((stage, i) => (
              <JourneyStage key={stage.id} stage={stage} index={i} activeIndex={activeIndex} />
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
