import { useEffect } from 'react'
import { useGLTF } from '@react-three/drei'

export function ModelInspector() {
  const gltf = useGLTF('/ball.glb')
  
  useEffect(() => {
    console.log('=== BALL.GLB INSPECTION ===')
    console.log('Full GLTF:', gltf)
    console.log('Scene:', gltf.scene)
    console.log('Animations:', gltf.animations)
    
    gltf.scene.traverse((child) => {
      console.log('Child:', child.name, child.type)
      
      if (child.isMesh) {
        console.log('  - Mesh found:', child.name)
        console.log('  - Geometry:', child.geometry)
        console.log('  - Material:', child.material)
        
        if (child.morphTargetDictionary) {
          console.log('  - Morph Targets:', child.morphTargetDictionary)
          console.log('  - Morph Influences:', child.morphTargetInfluences)
        }
      }
      
      if (child.isSkinnedMesh) {
        console.log('  - Skinned Mesh found!')
        console.log('  - Skeleton:', child.skeleton)
      }
    })
  }, [gltf])
  
  return null
}
