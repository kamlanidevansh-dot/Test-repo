"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { Line, PerspectiveCamera } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";

type JourneySceneProps = {
  progress: number;
  activeIndex: number;
  reducedMotion: boolean;
};

function AnimatedPath({ progress, activeIndex }: { progress: number; activeIndex: number }) {
  const group = useRef<THREE.Group>(null);

  // One continuous curve; stage transitions are represented by local curvature + glow intensity changes.
  const points = useMemo(
    () =>
      Array.from({ length: 220 }, (_, i) => {
        const t = i / 219;
        const amp = 0.28 + activeIndex * 0.04;
        return new THREE.Vector3(
          Math.sin(t * Math.PI * (1.6 + progress * 1.8)) * amp,
          2.7 - t * 5.4,
          Math.cos(t * Math.PI * 2.2) * (0.4 + progress * 0.65)
        );
      }),
    [progress, activeIndex]
  );

  const drawCount = Math.max(8, Math.floor(points.length * (0.15 + progress * 0.85)));
  const visible = points.slice(0, drawCount);

  useFrame((state) => {
    if (!group.current) return;
    group.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.18) * 0.22;
    group.current.rotation.x = -0.1 + Math.cos(state.clock.elapsedTime * 0.14) * 0.04;
  });

  return (
    <group ref={group}>
      <Line points={points} color="#ff522f" lineWidth={1} transparent opacity={0.12 + progress * 0.2} />
      <Line points={visible} color="#ff643e" lineWidth={3.6} transparent opacity={0.82} />
    </group>
  );
}

function Nodes({ progress, activeIndex }: { progress: number; activeIndex: number }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const temp = useMemo(() => new THREE.Object3D(), []);

  useFrame((state) => {
    if (!mesh.current) return;
    const count = 180;
    for (let i = 0; i < count; i += 1) {
      const t = i / count;
      const chaos = 1 - Math.min(progress * 1.2, 1);
      const radius = 0.35 + t * (0.4 + activeIndex * 0.06);
      temp.position.set(
        Math.sin(t * 40 + state.clock.elapsedTime * (0.2 + chaos * 0.8)) * radius,
        (0.5 - t) * 6,
        Math.cos(t * 30 + i) * (radius * (1.2 - chaos * 0.5))
      );
      temp.scale.setScalar(0.006 + progress * 0.014 + (i % 7 === 0 ? 0.012 : 0));
      temp.updateMatrix();
      mesh.current.setMatrixAt(i, temp.matrix);
    }
    mesh.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, 180]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshBasicMaterial color={activeIndex > 2 ? "#ff785a" : "#a863ff"} transparent opacity={0.48} />
    </instancedMesh>
  );
}

export function JourneyScene({ progress, activeIndex, reducedMotion }: JourneySceneProps) {
  return (
    <div className="h-[70vh] w-full md:h-[88vh]">
      <Canvas dpr={[1, 1.5]} gl={{ antialias: true, alpha: true }}>
        <PerspectiveCamera makeDefault position={[0, 0, 6]} fov={52} />
        <ambientLight intensity={0.6} />
        <pointLight color="#ff5f3a" intensity={8} distance={12} position={[1.5, 1.3, 3]} />
        <pointLight color="#a945ff" intensity={4.5} distance={10} position={[-1.3, -1.8, 2.5]} />
        <fog attach="fog" args={["#08080f", 4.2, 9.8]} />
        <AnimatedPath progress={progress} activeIndex={activeIndex} />
        {!reducedMotion && <Nodes progress={progress} activeIndex={activeIndex} />}
      </Canvas>
    </div>
  );
}
