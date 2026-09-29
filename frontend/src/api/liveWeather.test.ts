import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fetchLiveStationWeather } from './liveWeather'

describe('liveWeather API', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('successfully fetches and normalizes weather data from OpenWeather or Open-Meteo', async () => {
    const mockOpenWeatherResponse = {
      name: 'Puri Observation Buoy',
      sys: { country: 'IN' },
      coord: { lat: 18.24, lon: 86.85 },
      main: {
        temp: 28.5,
        feels_like: 32.1,
        humidity: 78,
        pressure: 1011,
      },
      wind: {
        speed: 4.5,
        deg: 180,
      },
      clouds: { all: 40 },
      weather: [{ description: 'moderate rain', icon: '10d' }],
      dt: 1755500000,
    }

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => mockOpenWeatherResponse,
    } as Response)

    const result = await fetchLiveStationWeather(18.24, 86.85)
    expect(result).not.toBeNull()
    expect(result?.stationName).toBe('Puri Observation Buoy')
    expect(result?.temp).toBe(28.5)
    expect(result?.humidity).toBe(78)
    expect(result?.source).toBe('OpenWeather')
  })

  it('falls back seamlessly to Open-Meteo when OpenWeather is unavailable', async () => {
    // 1st call fails (OpenWeather)
    vi.spyOn(globalThis, 'fetch')
      .mockRejectedValueOnce(new Error('OpenWeather rate limited'))
      // 2nd call succeeds (Open-Meteo)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          latitude: 18.24,
          longitude: 86.85,
          current: {
            temperature_2m: 29.0,
            apparent_temperature: 33.5,
            relative_humidity_2m: 82,
            surface_pressure: 1009.5,
            wind_speed_10m: 14.4, // 14.4 km/h = 4.0 m/s
            wind_direction_10m: 195,
            cloud_cover: 65,
          },
        }),
      } as Response)

    const result = await fetchLiveStationWeather(18.24, 86.85)
    expect(result).not.toBeNull()
    expect(result?.source).toBe('Open-Meteo')
    expect(result?.temp).toBe(29.0)
    expect(result?.humidity).toBe(82)
    expect(result?.windSpeed).toBe(4.0)
  })
})
