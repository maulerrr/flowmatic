/**
 * Generate semi-synthetic Astana traffic CSV for local Docker / thesis demos.
 * Schema matches services/sensor-simulator and models/flowml/data.py expectations.
 */
const fs = require('fs')
const path = require('path')

const outPath = path.join(__dirname, '..', 'data', 'astana_synthetic_data.csv')
const rows = 30000
const vehicleTypes = ['Car', 'Bus', 'Truck', 'Motorcycle', 'Taxi']
const eventTypes = ['Normal', 'Congestion', 'Accident', 'Roadwork', 'Weather']
const severities = ['Low', 'Medium', 'High', 'Critical']

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)]
}

function pad(n, w = 6) {
  return String(n).padStart(w, '0')
}

const header =
  'eventId,sourceTimestamp,vehicleType,speedKmh,latitude,longitude,eventType,severity,trafficDensity\n'
const start = new Date('2024-06-22T00:00:00.000Z').getTime()
const end = new Date('2024-09-30T23:59:59.000Z').getTime()

const lines = [header]
for (let i = 0; i < rows; i++) {
  const ts = new Date(start + Math.floor(Math.random() * (end - start))).toISOString()
  const latitude = Number((51.08 + Math.random() * 0.18).toFixed(6))
  const longitude = Number((71.32 + Math.random() * 0.22).toFixed(6))
  const speedKmh = Math.floor(5 + Math.random() * 95)
  const trafficDensity = Math.floor(10 + Math.random() * 190)
  lines.push(
    [
      `AST-${pad(i + 1)}`,
      ts,
      pick(vehicleTypes),
      speedKmh,
      latitude,
      longitude,
      pick(eventTypes),
      pick(severities),
      trafficDensity,
    ].join(',') + '\n',
  )
}

fs.mkdirSync(path.dirname(outPath), { recursive: true })
fs.writeFileSync(outPath, lines.join(''), 'utf8')
console.log(`Wrote ${rows} rows to ${outPath}`)
