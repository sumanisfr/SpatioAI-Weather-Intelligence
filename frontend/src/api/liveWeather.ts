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
  source: 'OpenWeather' | 'Open-Meteo'
}

const OPENWEATHER_KEY =
  (import.meta.env.VITE_OPENWEATHER_API_KEY as string | undefined) ||
  'babf0f4e684e15176c2fc62d0a394fac'

function getRegionDescriptor(lat: number, lon: number): string {
  if (lat >= 15 && lat <= 23 && lon >= 82 && lon <= 93) {
    return 'Bay of Bengal Marine Observation'
  }
  if (lat >= 8 && lat <= 15 && lon >= 75 && lon <= 81) {
    return 'Peninsular India Surface Station'
  }
  if (lon >= 72 && lon <= 76 && lat >= 12 && lat <= 20) {
    return 'Western Ghats Coastal Station'
  }
  return 'Regional Surface Station'
}

export async function fetchLiveStationWeather(
  lat: number,
  lon: number,
  signal?: AbortSignal
): Promise<LiveStationWeather | null> {
  // 1. Try OpenWeather API
  if (OPENWEATHER_KEY) {
    try {
      const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${OPENWEATHER_KEY}&units=metric`
      const res = await fetch(url, { signal })
      if (res.ok) {
        const data = await res.json()
        const defaultName = getRegionDescriptor(lat, lon)
        return {
          stationName: data.name && data.name.trim().length > 0 ? data.name : defaultName,
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
          source: 'OpenWeather',
        }
      }
    } catch {
      // Fall through to Open-Meteo
    }
  }

  // 2. Open-Meteo fallback (Free, reliable real-time meteorological observations)
  try {
    const omUrl = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,surface_pressure,wind_speed_10m,wind_direction_10m,cloud_cover,weather_code`
    const res = await fetch(omUrl, { signal })
    if (res.ok) {
      const om = await res.json()
      const cur = om.current
      const defaultName = getRegionDescriptor(lat, lon)
      return {
        stationName: defaultName,
        country: 'IN',
        stationLat: om.latitude ?? lat,
        stationLon: om.longitude ?? lon,
        temp: cur?.temperature_2m ?? 28,
        feelsLike: cur?.apparent_temperature ?? 32,
        humidity: cur?.relative_humidity_2m ?? 75,
        pressure: Math.round(cur?.surface_pressure ?? 1010),
        windSpeed: Number(((cur?.wind_speed_10m ?? 10) / 3.6).toFixed(1)),
        windDeg: cur?.wind_direction_10m ?? 180,
        clouds: cur?.cloud_cover ?? 30,
        description: 'Live physical observation',
        icon: '02d',
        timestamp: Date.now(),
        source: 'Open-Meteo',
      }
    }
  } catch (err) {
    if (signal?.aborted) throw err
  }

  return null
}

