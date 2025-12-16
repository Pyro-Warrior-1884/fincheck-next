"use client"

import ChartWrapper from "../ChartWrapper"

export default function ColdWarmBarChart({
  data,
}: {
  data: any[]
}) {
  const cold = data.filter((d) => d.cold_start).length
  const warm = data.length - cold

  return (
    <ChartWrapper
      title="Cold vs Warm Start Distribution"
      description="Cold starts significantly impact real-world latency"
    >
      <div className="flex gap-6 text-sm">
        <span className="text-red-600">
          Cold Starts: {cold}
        </span>
        <span className="text-green-600">
          Warm Starts: {warm}
        </span>
      </div>
    </ChartWrapper>
  )
}
