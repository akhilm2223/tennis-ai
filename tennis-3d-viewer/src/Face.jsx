import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'

export function Face({ isTalking, mouthOpenness = 0 }) {
  const upperLipRef = useRef()
  const lowerLipRef = useRef()
  const tongueRef = useRef()

  useFrame(() => {
    // EXTREME mouth movement - impossible to miss
    const maxSeparation = 0.5
    const separation = mouthOpenness * maxSeparation
    
    if (upperLipRef.current) {
      // Move upper lip UP dramatically
      upperLipRef.current.position.y = -0.05 + separation
      // Make it wider
      upperLipRef.current.scale.x = 1 + mouthOpenness * 1.5
      upperLipRef.current.scale.y = 1 + mouthOpenness * 0.5
      
      // Debug log every 30 frames
      if (Math.random() < 0.03) {
        console.log('👄 Mouth openness:', mouthOpenness.toFixed(2), 'Upper lip Y:', upperLipRef.current.position.y.toFixed(3))
      }
    }
    
    if (lowerLipRef.current) {
      // Move lower lip DOWN dramatically
      lowerLipRef.current.position.y = -0.05 - separation
      // Make it wider
      lowerLipRef.current.scale.x = 1 + mouthOpenness * 1.5
      lowerLipRef.current.scale.y = 1 + mouthOpenness * 0.5
    }
    
    // Show tongue when mouth opens
    if (tongueRef.current) {
      tongueRef.current.visible = mouthOpenness > 0.15
      tongueRef.current.scale.y = mouthOpenness * 2
    }
  })

  return (
    <group position={[0, 0, 0.27]}>
      {/* Left Eye */}
      <mesh position={[-0.08, 0.08, 0]}>
        <sphereGeometry args={[0.045, 16, 16]} />
        <meshStandardMaterial color="#000000" />
      </mesh>
      
      {/* Right Eye */}
      <mesh position={[0.08, 0.08, 0]}>
        <sphereGeometry args={[0.045, 16, 16]} />
        <meshStandardMaterial color="#000000" />
      </mesh>
    </group>
  )
}
