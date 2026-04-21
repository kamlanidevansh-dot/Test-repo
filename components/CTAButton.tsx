import { motion } from "framer-motion";
import Link from "next/link";

type CTAButtonProps = {
  href: string;
  children: React.ReactNode;
  variant?: "primary" | "secondary";
};

export function CTAButton({ href, children, variant = "primary" }: CTAButtonProps) {
  const isPrimary = variant === "primary";
  return (
    <motion.div whileHover={{ y: -2 }} whileTap={{ scale: 0.98 }}>
      <Link
        href={href}
        className={`inline-flex items-center justify-center rounded-full border px-6 py-3 text-sm font-semibold tracking-wide transition ${
          isPrimary
            ? "border-ember bg-gradient-to-r from-[#ff4d2d] to-[#ff6a3d] text-white shadow-ember"
            : "border-white/25 bg-white/5 text-white hover:bg-white/10"
        }`}
      >
        {children}
      </Link>
    </motion.div>
  );
}
