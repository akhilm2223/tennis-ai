import { useRef, forwardRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { useGLTF } from '@react-three/drei'
import { Face } from './Face'

export const CoachQ = forwardRef(({ isTalking, mouthOpenness = 0 }, ref) => {
  const { scene } = useGLTF('/ball.glb')
  const modelRef = useRef()
  const groupRef = useRef()

  useFrame(() => {
    if (modelRef.current && isTalking) {
      // Pulse ball size based on mouth openness
      const baseScale = 0.5
      const scaleAmount = mouthOpenness * 0.05
      const scale = baseScale + scaleAmount
      
      modelRef.current.scale.set(scale, scale, scale)
    } else if (modelRef.current) {
      modelRef.current.scale.set(0.5, 0.5, 0.5)
    }
  })

  return (
    <group ref={groupRef}>
      {/* Spotlight on the ball for subtle brightness */}
      <spotLight 
        position={[0, 0, 1]} 
        angle={0.5} 
        penumbra={0.5} 
        intensity={0.8}
        target-position={[0, 0, 0]}
      />
      
      {/* Tennis Ball - original colors from GLB with extra light */}
      <primitive 
        ref={modelRef}
        object={scene.clone()} 
        scale={0.5}
        position={[0, 0, 0]}
      />
      
      {/* Animated Face with AUDIO-DRIVEN Mouth */}
      <Face isTalking={isTalking} mouthOpenness={mouthOpenness} />
    </group>
  )
})
