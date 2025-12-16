"use client"

import CpuChart from "./charts/cpu/CpuChart"
import CpuRamBarChart from "./charts/cpu/CpuRamBarChart"
import ColdWarmBarChart from "./charts/coldwarm/ColdWarmBarChart"
import EntropyConfidenceBarChart from "./charts/entropy/EntropyConfidenceBarChart"
import OverallModelPerformanceBarChart from "./charts/overall/OverallModelPerformanceBarChart"
import LatencyLive from "./charts/latency/LatencyLive"
import CpuTimeSeriesChart from "./charts/cpu/CpuTimeSeriesChart"
import EnergyProxyBarChart from "./charts/overall/EnergyProxyBarChart"


type ChartItem = {
  model: string
  confidence: number
  entropy?: number
  latency_ms: number
  ram_mb: number
  cpu_percent?: number
  cold_start?: boolean
  score?: number
}

export default function ChartSection({
  data,
}: {
  data: ChartItem[]
}) {
  return (
    <div className="space-y-10">
      {/* RESOURCE USAGE */}
      <div className="grid gap-6 md:grid-cols-2">
        <CpuChart data={data} />
        <CpuRamBarChart data={data} />
      </div>

      {/* UNCERTAINTY */}
      <EntropyConfidenceBarChart data={data} />

      {/* COLD / WARM */}
      <ColdWarmBarChart data={data} />

      {/* TIME SERIES */}
      <div className="grid gap-6 md:grid-cols-2">
        <LatencyLive data={data} />
        <CpuTimeSeriesChart data={data} />
      </div>
      {/* 🔋 Energy Efficiency */}
<EnergyProxyBarChart data={data} />

      {/* OVERALL */}
      <OverallModelPerformanceBarChart data={data} />
    </div>
  )
}
