# Xceed Beyond — Premium Journey Experience

A single-page cinematic scroll experience built with **Next.js**, **Tailwind CSS**, **Framer Motion**, and **React Three Fiber**.

## Install

```bash
npm install
```

## Run locally

```bash
npm run dev
```

Open `http://localhost:3000`.

## Build

```bash
npm run build
npm run start
```

## Netlify build (uploadable)

This repo is configured for static export via Next.js (`output: "export"`), so the build output is generated in the `out/` folder.

```bash
npm install
npm run build
```

Then upload the **`out/` folder** to Netlify (or connect the repo and keep `netlify.toml` defaults).

### If uploading manually in Netlify UI

- Site settings → Deploys → "Deploy manually"
- Drag and drop the generated `out/` directory

### If connecting the repo

- Build command: `npm run build`
- Publish directory: `out`
- Node version: `20` (already set in `netlify.toml`)

## Architecture

- `app/page.tsx`: Main one-page experience (hero + sticky journey + scroll synchronization)
- `components/JourneyScene.tsx`: 3D/pseudo-3D evolving path + nodes
- `components/JourneyStage.tsx`: Reusable stage card with active/inactive states
- `components/CTAButton.tsx`: Reusable premium CTA button variants
- `lib/journeyData.ts`: Centralized copy and stage configuration

## Notes

- Desktop-first visual behavior with simplified motion when reduced-motion is enabled.
- The animated path evolves continuously with scroll from ambiguity to enterprise scale.
