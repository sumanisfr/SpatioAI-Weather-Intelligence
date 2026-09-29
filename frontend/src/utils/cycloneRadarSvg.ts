// High-fidelity Doppler Radar Cyclone Precipitation Field as an SVG Data URI
// Centered over Bay of Bengal and coastal Odisha (18.24°N, 86.85°E)

export const CYCLONE_BOUNDS: [[number, number], [number, number]] = [
  [12.0, 79.5], // South-West
  [24.5, 94.2], // North-East
]

const svgString = `
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 500" width="600" height="500">
  <defs>
    <radialGradient id="dopplerEye" cx="54%" cy="48%" r="48%">
      <stop offset="0%" stop-color="#7f1d1d" stop-opacity="0.98" />
      <stop offset="16%" stop-color="#dc2626" stop-opacity="0.95" />
      <stop offset="32%" stop-color="#ea580c" stop-opacity="0.92" />
      <stop offset="50%" stop-color="#f59e0b" stop-opacity="0.88" />
      <stop offset="68%" stop-color="#16a34a" stop-opacity="0.8" />
      <stop offset="84%" stop-color="#0891b2" stop-opacity="0.7" />
      <stop offset="94%" stop-color="#1d4ed8" stop-opacity="0.5" />
      <stop offset="100%" stop-color="#1e3a8a" stop-opacity="0.0" />
    </radialGradient>
    <filter id="radarGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3.5" />
    </filter>
  </defs>

  <g filter="url(#radarGlow)">
    <!-- Outer Deep Blue Feeder Band 1 -->
    <path
      d="M 180,410 C 260,460 390,430 480,330 C 540,260 550,170 480,110 C 400,40 280,70 220,150 C 180,210 200,280 260,310 C 310,320 370,290 380,230"
      fill="none"
      stroke="#1d4ed8"
      stroke-width="38"
      stroke-linecap="round"
      opacity="0.6"
    />

    <!-- Cyan Outer Convective Band 2 -->
    <path
      d="M 210,380 C 280,420 380,400 450,310 C 500,250 510,180 450,130 C 380,70 280,100 230,165 C 200,215 220,270 270,295"
      fill="none"
      stroke="#0891b2"
      stroke-width="30"
      stroke-linecap="round"
      opacity="0.7"
    />

    <!-- Emerald Rain Band -->
    <path
      d="M 235,340 C 290,380 370,360 425,290 C 465,240 470,190 420,150 C 365,100 285,120 245,180 C 220,220 235,260 275,275"
      fill="none"
      stroke="#16a34a"
      stroke-width="24"
      stroke-linecap="round"
      opacity="0.8"
    />

    <!-- Yellow Intense Convective Band -->
    <path
      d="M 255,305 C 300,340 360,325 400,270 C 435,225 435,185 395,155 C 355,120 295,135 265,185 C 245,215 255,245 285,255"
      fill="none"
      stroke="#f59e0b"
      stroke-width="20"
      stroke-linecap="round"
      opacity="0.88"
    />

    <!-- Orange-Red Eyewall Outer Spiral -->
    <path
      d="M 275,275 C 310,305 350,290 380,250 C 405,215 405,185 375,165 C 345,145 305,155 285,190 C 270,215 280,235 300,240"
      fill="none"
      stroke="#ea580c"
      stroke-width="18"
      stroke-linecap="round"
      opacity="0.92"
    />

    <!-- High-Intensity Eyewall Core Envelope -->
    <ellipse cx="325" cy="240" rx="110" ry="90" fill="url(#dopplerEye)" />

    <!-- Central Storm Eye -->
    <circle cx="325" cy="240" r="22" fill="#7f1d1d" />
    <circle cx="325" cy="240" r="14" fill="#dc2626" />
    <circle cx="325" cy="240" r="5" fill="#ffffff" />
  </g>
</svg>
`.trim()

export const CYCLONE_RADAR_SVG_DATA_URL = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgString)}`
