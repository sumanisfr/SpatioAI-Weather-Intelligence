export type LiveStationWeather = {
  stationName: string
  country: string
  stationLat: number
  stationLon: number
  temp: number
  feelsLike: number
  humidity: number
  pressure: number
  windSpeed: number
  windDeg: number
  clouds: number
  description: string
  icon: string
  timestamp: number
}

const OPENWEATHER_KEY = (import.meta.env.VITE_OPENWEATHER_API_KEY as string | undefined) || 'babf0f4e684e15176c2fc62d0a394fac'

export async function fetchLiveStationWeather(lat: number, lon: number, signal?: AbortSignal): Promise<LiveStationWeather | null> {
  if (!OPENWEATHER_KEY) return null
  const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${OPENWEATHER_KEY}&units=metric`
  const res = await fetch(url, { signal })
  if (!res.ok) {
    throw new Error(`OpenWeather API status: ${res.statusText}`)
  }
  const data = await res.json()
  return {
    stationName: data.name || 'Regional Station',
    country: data.sys?.country || 'IN',
    stationLat: data.coord?.lat ?? lat,
    stationLon: data.coord?.lon ?? lon,
    temp: data.main?.temp ?? 0,
    feelsLike: data.main?.feels_like ?? 0,
    humidity: data.main?.humidity ?? 0,
    pressure: data.main?.pressure ?? 0,
    windSpeed: data.wind?.speed ?? 0,
    windDeg: data.wind?.deg ?? 0,
    clouds: data.clouds?.all ?? 0,
    description: data.weather?.[0]?.description ?? 'Current conditions',
    icon: data.weather?.[0]?.icon ?? '03d',
    timestamp: (data.dt ?? Math.floor(Date.now() / 1000)) * 1000,
  }
}
